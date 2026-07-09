"""ASL(수어 지문자) 이미지를 업로드하면 알파벳을 예측하는 간단한 Streamlit 앱.

state_dict 방식으로 학습된 가중치만 로드하므로, 아래 build_model()에
학습 때와 동일한 모델 구조를 정의해 두어야 한다.
"""

import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms
import pandas as pd
from PIL import Image

from utils import MyConvBlock  # 모델 구조에 쓰이는 커스텀 Conv 블록


# =====================
# 설정
# =====================
MODEL_PATH = "model/asl_model_state.pth"  # state_dict(가중치만) 저장 파일
IMG_SIZE = (28, 28)                       # 학습 이미지 크기
IMG_CHS = 1                               # 입력 채널 수 (흑백)
N_CLASSES = 24                            # Sign-MNIST 클래스 수 (동작이 필요한 J, Z 제외)
TOP_K = 3                                 # 상위 몇 개 예측을 보여줄지

# 클래스 인덱스 → 알파벳 (J, Z 제외한 24자)
CLASS_NAMES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================
# 모델
# =====================
def build_model():
    """학습 때와 동일한 CNN 구조를 새 인스턴스로 만든다. (입력: 1 x 28 x 28)"""
    return nn.Sequential(
        MyConvBlock(IMG_CHS, 25, 0.0),   # -> 25 x 14 x 14
        MyConvBlock(25, 50, 0.2),        # -> 50 x 7 x 7
        MyConvBlock(50, 75, 0.0),        # -> 75 x 3 x 3
        nn.Flatten(),
        nn.Linear(75 * 3 * 3, 512),
        nn.Dropout(0.3),
        nn.ReLU(),
        nn.Linear(512, N_CLASSES),
    )


@st.cache_resource(show_spinner=False)  # 모델을 한 번만 로드해 재사용
def load_model(path):
    """가중치(state_dict)를 읽어 모델에 적용하고 추론 모드로 반환한다."""
    model = build_model().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


# 학습이 [0,1] 스케일만 했으므로 추론도 동일하게 맞춘다.
preprocess = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.PILToTensor(),                      # -> C x H x W (uint8)
    transforms.ToDtype(torch.float32, scale=True), # [0,255] -> [0,1]
])


# =====================
# 화면
# =====================
st.set_page_config(page_title="Simple ASL Prediction App", page_icon="🤟", layout="centered")
st.title("🤟 Simple ASL Prediction App")

try:
    model = load_model(MODEL_PATH)
    st.caption(f"모델 로드 완료: `{MODEL_PATH}`")
except Exception as e:
    st.error(f"모델 로드 실패: {e}")
    st.stop()

# 이미지 업로드 & 미리보기
file = st.file_uploader("테스트 이미지 업로드 (png/jpg 등)", type=["png", "jpg", "jpeg", "bmp", "webp"])

image = None
if file is not None:
    image = Image.open(file).convert("L")  # 어떤 포맷이든 1채널 흑백으로 통일
    st.image(image, caption="업로드한 이미지", width=image.width * 2)

# 예측
if st.button("예측하기", disabled=(image is None)):
    with st.spinner("예측 중..."):
        x = preprocess(image).unsqueeze(0).to(device)  # [1, 1, 28, 28]
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1).squeeze(0)  # [24]
            top_probs, top_idxs = torch.topk(probs, min(TOP_K, probs.numel()))

    # 1위는 강조, 전체 상위 K개는 표 + 막대 차트로
    labels = [CLASS_NAMES[int(i)] for i in top_idxs]
    percents = [round(float(p) * 100, 2) for p in top_probs]

    st.success(f"예측 결과: {labels[0]} ({percents[0]:.2f}%)")

    df = pd.DataFrame({"확률(%)": percents}, index=labels)
    df.index.name = "클래스"
    st.markdown(f"**상위 {len(labels)}개 예측**")
    st.bar_chart(df, y="확률(%)")
    st.dataframe(df, width="stretch")

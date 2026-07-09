"""ASL(수어 지문자) 이미지를 업로드/촬영하면 알파벳을 예측하는 Streamlit 앱.

Sign-MNIST(28x28 흑백, 손 중앙 정렬)로 학습된 CNN이라, 스마트폰 실사진과는
분포 차이가 크다. 아래 preprocess_photo()가 실사진을 학습 분포에 최대한 맞춰준다.
state_dict 방식이라 build_model()에 학습 때와 동일한 구조를 정의해 둔다.
"""

import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms
import pandas as pd
from PIL import Image, ImageOps

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


# =====================
# 전처리 (스마트폰 실사진 → 학습 분포에 맞추기)
# =====================
# 학습이 x/255 로 [0,1] 스케일만 했으므로 텐서 변환도 [0,1] 까지만.
to_tensor = transforms.Compose([
    transforms.PILToTensor(),                      # -> C x H x W (uint8)
    transforms.ToDtype(torch.float32, scale=True), # [0,255] -> [0,1]
])


def preprocess_photo(img, invert=False):
    """임의의 사진을 28x28 흑백, 중앙 정렬로 변환해 학습 분포에 가깝게 만든다."""
    img = ImageOps.exif_transpose(img)          # 스마트폰 회전(EXIF) 보정
    img = img.convert("L")                       # 1채널 흑백

    # 짧은 변 기준으로 중앙 정사각형 크롭 → 비율 왜곡 없이 정사각형 확보
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    img = img.crop((left, top, left + s, top + s))

    img = ImageOps.autocontrast(img, cutoff=2)   # 명암 스트레칭으로 조명 편차 완화
    if invert:
        img = ImageOps.invert(img)               # 밝기 극성 반전
    return img.resize(IMG_SIZE, Image.BILINEAR)  # 28x28로 축소


def predict_probs(img28):
    """28x28 흑백 PIL 이미지 하나에 대한 클래스 확률 [24] 반환."""
    x = to_tensor(img28).unsqueeze(0).to(device)  # [1, 1, 28, 28]
    with torch.no_grad():
        return torch.softmax(model(x), dim=1).squeeze(0)


def predict_best(img, auto_polarity=True, invert=False):
    """전처리 + 예측. auto_polarity면 정상/반전 둘 다 추론해 더 확신 높은 쪽 채택.

    반환: (확률 [24], 사용한 28x28 이미지, 반전 사용 여부)
    """
    if auto_polarity:
        img_a = preprocess_photo(img, invert=False)
        img_b = preprocess_photo(img, invert=True)
        probs_a, probs_b = predict_probs(img_a), predict_probs(img_b)
        # top-1 확률이 더 높은(=더 확신하는) 극성을 선택
        if probs_b.max() > probs_a.max():
            return probs_b, img_b, True
        return probs_a, img_a, False
    img28 = preprocess_photo(img, invert=invert)
    return predict_probs(img28), img28, invert


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

st.info(
    "📸 **정확도 팁**: 손을 화면 **중앙에 크게**, **단색 배경**, **밝은 조명**에서 촬영하세요.\n\n"
    "이 모델은 28×28 흑백 데이터(Sign-MNIST)로 학습돼, 아래 '모델 입력' 미리보기처럼 "
    "손 모양이 또렷할수록 잘 맞힙니다."
)

# 입력: 카메라 촬영 또는 파일 업로드
tab_cam, tab_upload = st.tabs(["📷 카메라 촬영", "📁 파일 업로드"])
with tab_cam:
    cam_img = st.camera_input("손 모양을 중앙에 두고 촬영하세요")
with tab_upload:
    up_img = st.file_uploader("이미지 업로드 (png/jpg 등)", type=["png", "jpg", "jpeg", "bmp", "webp"])

raw = cam_img or up_img

with st.expander("⚙️ 고급 설정"):
    auto_polarity = st.checkbox(
        "밝기 극성 자동 선택", value=True,
        help="정상/반전 두 버전을 모두 추론해 더 확신하는 쪽을 자동 선택합니다.",
    )
    manual_invert = st.checkbox(
        "수동 색 반전", value=False, disabled=auto_polarity,
        help="자동 선택을 끈 경우에만 적용됩니다.",
    )

# 예측
if raw is not None:
    src = Image.open(raw)
    probs, img28, used_invert = predict_best(src, auto_polarity, manual_invert)
    top_probs, top_idxs = torch.topk(probs, min(TOP_K, probs.numel()))

    # 원본 vs 모델이 실제 보는 28x28 입력을 나란히 표시
    col1, col2 = st.columns(2)
    with col1:
        st.image(ImageOps.exif_transpose(src), caption="입력 이미지", use_container_width=True)
    with col2:
        # 28x28은 너무 작으니 확대해서 보여줌
        st.image(img28.resize((196, 196), Image.NEAREST),
                 caption=f"모델 입력 (28×28{', 반전' if used_invert else ''})",
                 use_container_width=True)

    labels = [CLASS_NAMES[int(i)] for i in top_idxs]
    percents = [round(float(p) * 100, 2) for p in top_probs]

    st.success(f"예측 결과: {labels[0]} ({percents[0]:.2f}%)")
    if percents[0] < 50:
        st.warning("확신도가 낮습니다. 손을 더 크게·중앙에, 배경을 단순하게 다시 촬영해 보세요.")

    df = pd.DataFrame({"확률(%)": percents}, index=labels)
    df.index.name = "클래스"
    st.markdown(f"**상위 {len(labels)}개 예측**")
    st.bar_chart(df, y="확률(%)")
    st.dataframe(df, width="stretch")

"""ASL(수어 지문자) 이미지를 업로드/촬영하면 알파벳을 예측하는 Streamlit 앱.

Sign-MNIST(28x28 흑백, 손 중앙 정렬)로 학습된 CNN이라 실사진과 분포 차이가 크다.
전처리로 그 간극을 좁힌다:
  1) MediaPipe로 손 영역을 찾아 확대 크롭 (없으면 중앙 크롭으로 폴백)
  2) 흑백 + autocontrast + 28x28 리사이즈(LANCZOS)
  3) 밝기 극성(정상/반전) 자동 선택
state_dict 방식이라 build_model()에 학습 때와 동일한 구조를 정의해 둔다.
"""

import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms
import pandas as pd
from PIL import Image, ImageOps

from utils import MyConvBlock  # 모델 구조에 쓰이는 커스텀 Conv 블록


# 설정
MODEL_PATH = "model/asl_model_state.pth"
IMG_SIZE = (28, 28)
N_CLASSES = 24
TOP_K = 3
HAND_PAD = 0.6                                   # 손 bbox 여백 비율
CLASS_NAMES = list("ABCDEFGHIKLMNOPQRSTUVWXY")   # J, Z 제외한 24자

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================
# 모델
# =====================
def build_model():
    """학습 때와 동일한 CNN 구조. (입력: 1 x 28 x 28)"""
    return nn.Sequential(
        MyConvBlock(1, 25, 0.0),   # -> 25 x 14 x 14
        MyConvBlock(25, 50, 0.2),  # -> 50 x 7 x 7
        MyConvBlock(50, 75, 0.0),  # -> 75 x 3 x 3
        nn.Flatten(),
        nn.Linear(75 * 3 * 3, 512),
        nn.Dropout(0.3),
        nn.ReLU(),
        nn.Linear(512, N_CLASSES),
    )


@st.cache_resource(show_spinner=False)  # 모델을 한 번만 로드해 재사용
def load_model():
    model = build_model().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model


# =====================
# 손 검출 (MediaPipe) — 없으면 자동 폴백
# =====================
@st.cache_resource(show_spinner=False)
def get_hand_detector():
    import mediapipe as mp
    return mp.solutions.hands.Hands(
        static_image_mode=True, max_num_hands=1,
        model_complexity=1, min_detection_confidence=0.5,
    )


def detect_hand_bbox(img):
    """손 랜드마크로 여백 포함 정사각형 bbox(픽셀)를 계산. 실패/미검출/미설치 시 None."""
    try:
        import numpy as np
        rgb = np.array(img.convert("RGB"))
        result = get_hand_detector().process(rgb)
    except Exception:
        return None
    if not result.multi_hand_landmarks:
        return None

    h, w = rgb.shape[:2]
    xs = [p.x * w for p in result.multi_hand_landmarks[0].landmark]  # 픽셀 좌표
    ys = [p.y * h for p in result.multi_hand_landmarks[0].landmark]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    half = max(max(xs) - min(xs), max(ys) - min(ys)) * (1 + HAND_PAD) / 2
    return (int(max(0, cx - half)), int(max(0, cy - half)),
            int(min(w, cx + half)), int(min(h, cy + half)))


# =====================
# 전처리 & 예측
# =====================
# 학습이 x/255 로 [0,1] 스케일만 했으므로 텐서 변환도 [0,1] 까지만.
to_tensor = transforms.Compose([
    transforms.PILToTensor(),
    transforms.ToDtype(torch.float32, scale=True),  # [0,255] -> [0,1]
])


def to_square_gray(img):
    """사진을 정사각형 흑백으로 자른다. 손 검출 성공 시 손 영역, 아니면 중앙 크롭.

    반환: (정사각 흑백 PIL, 손 검출 여부)
    """
    img = ImageOps.exif_transpose(img)          # 스마트폰 회전(EXIF) 보정
    box = detect_hand_bbox(img)
    gray = img.convert("L")
    if box is None:                             # 중앙 정사각형 크롭(폴백)
        w, h = gray.size
        s = min(w, h)
        box = ((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)
        return gray.crop(box), False
    return gray.crop(box), True


def to_model_input(gray_square, invert):
    """정사각 흑백 → autocontrast(+반전) → 28x28. 고해상도 대비 LANCZOS로 안티앨리어싱."""
    g = ImageOps.autocontrast(gray_square, cutoff=2)
    if invert:
        g = ImageOps.invert(g)
    return g.resize(IMG_SIZE, Image.Resampling.LANCZOS)


def predict(img):
    """전처리 후 정상/반전 둘 다 추론해 더 확신 높은 극성을 채택.

    반환: (확률 [24], 사용한 28x28 이미지, 반전 여부, 손 검출 여부)
    """
    gray_sq, detected = to_square_gray(img)
    candidates = []
    for invert in (False, True):
        img28 = to_model_input(gray_sq, invert)
        x = to_tensor(img28).unsqueeze(0).to(device)  # [1, 1, 28, 28]
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1).squeeze(0)  # [24]
        candidates.append((probs.max().item(), probs, img28, invert))

    _, probs, img28, invert = max(candidates, key=lambda c: c[0])
    return probs, img28, invert, detected


# =====================
# 화면
# =====================
st.set_page_config(page_title="Simple ASL Prediction App", page_icon="🤟", layout="centered")
st.title("🤟 Simple ASL Prediction App")

model = load_model()

st.info("📸 손을 화면 **중앙에 크게**, **단색 배경**, **밝은 조명**에서 촬영하면 정확도가 올라갑니다.")

# 입력: 카메라 촬영 또는 파일 업로드
tab_cam, tab_upload = st.tabs(["📷 카메라 촬영", "📁 파일 업로드"])
with tab_cam:
    cam_img = st.camera_input("손 모양을 중앙에 두고 촬영하세요")
with tab_upload:
    up_img = st.file_uploader("이미지 업로드 (png/jpg 등)", type=["png", "jpg", "jpeg", "bmp", "webp"])

raw = cam_img or up_img

# 예측
if raw is not None:
    with st.spinner("예측 중..."):
        probs, img28, used_invert, detected = predict(Image.open(raw))
    top_probs, top_idxs = torch.topk(probs, min(TOP_K, probs.numel()))

    crop_note = "손 감지 크롭" if detected else "중앙 크롭"
    caption = f"모델 입력 (28×28 · {crop_note}{' · 반전' if used_invert else ''})"

    # 원본 vs 모델이 실제 보는 28x28 입력을 나란히 표시
    col1, col2 = st.columns(2)
    col1.image(ImageOps.exif_transpose(Image.open(raw)), caption="입력 이미지", use_container_width=True)
    col2.image(img28.resize((196, 196), Image.Resampling.NEAREST), caption=caption, use_container_width=True)

    labels = [CLASS_NAMES[int(i)] for i in top_idxs]
    percents = [round(float(p) * 100, 2) for p in top_probs]

    st.success(f"예측 결과: {labels[0]} ({percents[0]:.2f}%)")
    if percents[0] < 50:
        st.warning("확신도가 낮습니다. 손을 더 크게·중앙에, 배경을 단순하게 다시 촬영해 보세요.")

    df = pd.DataFrame({"확률(%)": percents}, index=labels)
    df.index.name = "클래스"
    st.bar_chart(df, y="확률(%)")

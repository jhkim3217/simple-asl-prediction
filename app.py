"""ASL(수어 지문자) 이미지를 업로드/촬영하면 알파벳을 예측하는 Streamlit 앱.

Sign-MNIST(28x28 흑백, 손 중앙 정렬)로 학습된 CNN이라, 스마트폰 실사진과는
분포 차이가 크다. 아래 전처리가 실사진을 학습 분포에 최대한 맞춰준다:
  1) MediaPipe로 손 영역을 찾아 확대 크롭 (있으면)  → 학습 데이터처럼 손이 프레임에 꽉 참
  2) 없으면 중앙 정사각형 크롭으로 폴백
  3) 흑백 + autocontrast + 28x28 리사이즈
  4) 밝기 극성(정상/반전) 자동 선택
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
HAND_PAD = 0.6                            # 손 bbox 여백 비율 (손목/여유 포함)

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
# 손 검출 (MediaPipe) — 없으면 안전하게 폴백
# =====================
@st.cache_resource(show_spinner=False)
def get_hand_detector():
    """MediaPipe Hands 인스턴스를 한 번만 만들어 재사용. 미설치 시 예외 → 호출부에서 폴백."""
    import mediapipe as mp
    return mp.solutions.hands.Hands(
        static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3
    )


def hand_crop_available():
    """이 환경에서 MediaPipe를 쓸 수 있는지 확인 (aarch64 등에는 휠이 없을 수 있음)."""
    try:
        import mediapipe  # noqa: F401
        return True
    except Exception:
        return False


def detect_hand_bbox(pil_img, pad=HAND_PAD):
    """손 랜드마크로 정사각형 bbox(픽셀)를 계산. 실패/미검출 시 None."""
    try:
        import numpy as np
        detector = get_hand_detector()
        rgb = np.array(pil_img.convert("RGB"))
        h, w = rgb.shape[:2]
        result = detector.process(rgb)
    except Exception:
        return None
    if not result.multi_hand_landmarks:
        return None

    lm = result.multi_hand_landmarks[0].landmark
    xs = [p.x for p in lm]  # 정규화 좌표 [0,1]
    ys = [p.y for p in lm]
    x0, x1 = min(xs) * w, max(xs) * w
    y0, y1 = min(ys) * h, max(ys) * h

    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0) * (1 + pad) / 2   # 여백 포함 정사각 반변
    l, t = max(0, cx - half), max(0, cy - half)
    r, b = min(w, cx + half), min(h, cy + half)
    if r - l < 8 or b - t < 8:
        return None
    return (int(l), int(t), int(r), int(b))


# =====================
# 전처리 & 예측
# =====================
# 학습이 x/255 로 [0,1] 스케일만 했으므로 텐서 변환도 [0,1] 까지만.
to_tensor = transforms.Compose([
    transforms.PILToTensor(),                      # -> C x H x W (uint8)
    transforms.ToDtype(torch.float32, scale=True), # [0,255] -> [0,1]
])


def prepare_square(img, use_hand_crop):
    """사진을 정사각형 흑백으로 자른다. 손 검출 성공 시 손 영역, 아니면 중앙 크롭.

    반환: (정사각 흑백 PIL, 손 검출 여부)
    """
    img = ImageOps.exif_transpose(img)                 # 스마트폰 회전(EXIF) 보정
    box = detect_hand_bbox(img) if use_hand_crop else None
    gray = img.convert("L")                            # 1채널 흑백
    if box:
        gray = gray.crop(box)                          # 손 영역 크롭
    else:
        w, h = gray.size                               # 중앙 정사각형 크롭(폴백)
        s = min(w, h)
        l, t = (w - s) // 2, (h - s) // 2
        gray = gray.crop((l, t, l + s, t + s))
    return gray, (box is not None)


def to_model_input(gray_square, invert):
    """정사각 흑백 → autocontrast(+반전) → 28x28 모델 입력 이미지."""
    g = ImageOps.autocontrast(gray_square, cutoff=2)   # 조명 편차 완화
    if invert:
        g = ImageOps.invert(g)
    return g.resize(IMG_SIZE, Image.BILINEAR)


def predict_probs(img28):
    """28x28 흑백 PIL 이미지 하나에 대한 클래스 확률 [24] 반환."""
    x = to_tensor(img28).unsqueeze(0).to(device)       # [1, 1, 28, 28]
    with torch.no_grad():
        return torch.softmax(model(x), dim=1).squeeze(0)


def predict_best(img, auto_polarity=True, invert=False, use_hand_crop=True):
    """손 크롭(1회) 후 예측. auto_polarity면 정상/반전 둘 다 추론해 더 확신 높은 쪽 채택.

    반환: (확률 [24], 사용한 28x28 이미지, 반전 사용 여부, 손 검출 여부)
    """
    gray_sq, detected = prepare_square(img, use_hand_crop)
    if auto_polarity:
        img_a = to_model_input(gray_sq, False)
        img_b = to_model_input(gray_sq, True)
        probs_a, probs_b = predict_probs(img_a), predict_probs(img_b)
        if probs_b.max() > probs_a.max():
            return probs_b, img_b, True, detected
        return probs_a, img_a, False, detected
    img28 = to_model_input(gray_sq, invert)
    return predict_probs(img28), img28, invert, detected


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

MP_OK = hand_crop_available()

st.info(
    "📸 **정확도 팁**: 손을 화면 **중앙에 크게**, **단색 배경**, **밝은 조명**에서 촬영하세요.\n\n"
    + ("✋ **손 자동 크롭(MediaPipe)** 이 켜져 있어, 사진 속 손을 찾아 확대해 인식합니다."
       if MP_OK else
       "⚠️ 이 환경엔 MediaPipe가 없어 **중앙 크롭**으로 동작합니다. (손을 중앙에 크게 두세요)")
)

# 입력: 카메라 촬영 또는 파일 업로드
tab_cam, tab_upload = st.tabs(["📷 카메라 촬영", "📁 파일 업로드"])
with tab_cam:
    cam_img = st.camera_input("손 모양을 중앙에 두고 촬영하세요")
with tab_upload:
    up_img = st.file_uploader("이미지 업로드 (png/jpg 등)", type=["png", "jpg", "jpeg", "bmp", "webp"])

raw = cam_img or up_img

with st.expander("⚙️ 고급 설정"):
    use_hand_crop = st.checkbox(
        "✋ 손 자동 크롭 (MediaPipe)", value=MP_OK, disabled=not MP_OK,
        help="사진에서 손 영역을 찾아 확대 크롭 → 학습 데이터처럼 손이 꽉 차게 만듭니다.",
    )
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
    with st.spinner("예측 중..."):
        probs, img28, used_invert, detected = predict_best(
            src, auto_polarity, manual_invert, use_hand_crop
        )
    top_probs, top_idxs = torch.topk(probs, min(TOP_K, probs.numel()))

    # 크롭 방식 안내 문구
    if use_hand_crop and detected:
        crop_note = "손 감지 → 자동 크롭"
    elif use_hand_crop:
        crop_note = "손 미감지 → 중앙 크롭"
    else:
        crop_note = "중앙 크롭"

    # 원본 vs 모델이 실제 보는 28x28 입력을 나란히 표시
    col1, col2 = st.columns(2)
    with col1:
        st.image(ImageOps.exif_transpose(src), caption="입력 이미지", use_container_width=True)
    with col2:
        st.image(img28.resize((196, 196), Image.NEAREST),  # 28x28은 너무 작으니 확대
                 caption=f"모델 입력 (28×28 · {crop_note}{' · 반전' if used_invert else ''})",
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

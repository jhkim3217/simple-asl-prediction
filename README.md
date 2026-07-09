# 🤟 Simple ASL Prediction App

ASL(미국 수어) 지문자 이미지를 업로드하면 알파벳을 예측하는 Streamlit + PyTorch 앱입니다.

- **분류 대상**: A–Y 24개 알파벳 (동작이 필요한 J, Z 제외 — Sign-MNIST 기준)
- **모델**: 3-block CNN (Conv + BatchNorm + Dropout + MaxPool) + FC
- **로딩 방식**: `state_dict`(가중치만) 로드 — torch 버전에 강건
- **학습 환경**: NVIDIA DGX Spark (Grace Blackwell GB10)

## 로컬 실행

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 배포 (Streamlit Community Cloud)

1. [share.streamlit.io](https://share.streamlit.io) 접속 → GitHub 연결
2. 이 레포지토리 선택, **Main file path** = `app.py`
3. Deploy 클릭

`test_image/`의 `a.png`, `b.png`로 바로 테스트할 수 있습니다.

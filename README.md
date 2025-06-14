# MTST_Custom

# MTST_custom

 개요

본 저장소는 networkslab/MTST를 기반으로 사용자 정의 시계열 데이터 (Volatility) 예측 수행

사용 중 발생한 오류를 해결하고 안정성을 높인 사용자 커스터마이즈 버전

2. 라이선스 및 저작권 고지

원본 MTST는 MIT License로 배포됩니다.

본 저장소는 원본 라이선스를 준수하며, 원 저작권(networkslab)에 귀속됩니다.

사용자는 연구·비상업적 목적에 한해 자유롭게 사용할 수 있으며 상업적 이용 시 반드시 원저작자 라이선스를 따라야 합니다.

3. 주요 수정 내역

파일

수정 내용

| 수정 항목 | 상세 내용 | 목적 |
|-----------|-----------------------------|------------------------------|
| **Metric 저장 오류** | `exp/exp_main.py`에서 `corr` 배열 스칼라화 (`np.mean` 처리) | `metrics.npy` 저장 오류 해결 |
| **Split 크기 부족 오류** | `run_longExp.py`에 `seq_len + pred_len` 사전 검증 추가 | 학습/테스트 중 IndexError 사전 방지 |
| **`__len__` 로직 강화** | `data_provider/data_loader.py`에 상세 오류 메시지 + 안전한 pred_len 추천값 출력 | 사용자 디버깅 편의성 |
| **환경 설정** | `yacs` 미설치 시 `ModuleNotFoundError` 발생 → `pip install yacs` 필요 | 의존성 누락 방지 |

---

실제 수정 파일

| 파일 | 수정 위치 | 주요 변경 |
|------|------------|------------|
| `exp/exp_main.py` | `test()` | `corr` 스칼라화 후 numpy 저장 |
| `run_longExp.py` | training 시작 전 | split 크기 사전 검증 |
| `data_provider/data_loader.py` | `__len__` | split validation 메시지 개선 |
| Colab 환경 | - | `pip install yacs` 명시 |


4. 데이터 기준

전체 rows: 4924

Split 비율: train 70%, val 10%, test 20% (예시)

5. 추천 설정

seq_len + pred_len < 각 split 크기

예: pred_len=720 가능하려면 가장 작은 split 크기보다 작게 설정

6. 필수 의존성 설치

pip install yacs

7. 주요 코드 스니펫

# exp/exp_main.py
if isinstance(corr, (list, np.ndarray)):
    corr = np.mean(corr)
np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe, rse, corr]))

# run_longExp.py
train_data, _ = data_provider(args, 'train')
val_data, _ = data_provider(args, 'val')
test_data, _ = data_provider(args, 'test')
min_len = min(len(train_data), len(val_data), len(test_data))
if args.seq_len + args.pred_len > min_len:
    raise ValueError(
        f"Combined seq_len + pred_len exceeds split length. Adjust your settings.")

8. 실행 예시

python run_longExp.py \
  --is_training 1 \
  --model MTST \
  --data custom \
  --target Volatility \
  --seq_len 336 \
  --pred_len 720

'MTST.ipynb' 참고

9. 참고 자료

MIT License
자세한 사항은 LICENSE 파일 참조

데이터: DataGuide, FnGuide MK2000 지수

참고문헌: Zhang, Yitian, et al. "Multi-resolution time-series transformer for long-term forecasting." International conference on artificial intelligence and statistics. PMLR, 2024.

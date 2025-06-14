from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from torch.utils.data import DataLoader

# 데이터셋을 처리할 수 있는 데이터 로더 매핑
data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,  # 사용자 정의 데이터셋에 기본 매핑
}

def data_provider(args, flag):
    """
    데이터셋 및 데이터 로더를 반환하는 함수
    :param args: 스크립트에서 전달된 인자들
    :param flag: 'train', 'test', 'pred' 중 하나
    :return: 데이터셋 객체와 데이터 로더
    """
    # 사용자 정의 데이터 파일 처리 추가
    if args.data.endswith('.csv'):
        Data = Dataset_Custom  # 모든 CSV 파일은 Dataset_Custom으로 처리
    else:
        Data = data_dict.get(args.data, None)
        if Data is None:
            raise ValueError(f"Unknown dataset: {args.data}")

    # Time encoding 설정
    timeenc = 0 if args.embed != 'timeF' else 1

    # Flag에 따른 데이터 로더 설정
    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:  # train
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    # 데이터셋 객체 생성
    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq
    )
    print(f"{flag} set size: {len(data_set)}")

    # 데이터 로더 생성
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last
    )

    return data_set, data_loader


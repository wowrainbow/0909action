from tensorflow import keras
import argparse
import warnings
warnings.filterwarnings("ignore")

def dataloader(dataset_name='mnist'):
    """
    데이터셋을 불러오고 학습 및 테스트 데이터를 반환합니다.
    
    Args:
        dataset_name (str): 사용할 데이터셋의 이름 (기본값은 'mnist').
    
    Returns:
        (x_train, y_train), (x_test, y_test): 학습 및 테스트 데이터셋을 반환합니다.
    """
    
    # 사용할 수 있는 데이터셋을 매핑합니다.
    datasets = {
        'mnist': keras.datasets.mnist,
        'fashion_mnist': keras.datasets.fashion_mnist,
        'cifar10': keras.datasets.cifar10,
    }
    
    if dataset_name not in datasets:
        raise ValueError(f"지원되지 않는 데이터셋: {dataset_name}. 사용 가능한 데이터셋: {list(datasets.keys())}")
    
    dataset = datasets[dataset_name]
    
    # 데이터셋을 로드합니다.
    (x_train, y_train), (x_test, y_test) = dataset.load_data()   

    print(f"\n{dataset_name} 데이터셋 로드 완료")

    # 데이터 정규화 (0~255 범위의 픽셀 값을 0~1 범위로 변환)
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    return (x_train, y_train), (x_test, y_test)
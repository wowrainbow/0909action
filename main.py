from datasets import dataloader
from model import create_model
from training import training
import argparse

def main(dataset_name, epochs):
    # 1. 데이터셋 로드
    (x_train, y_train), (x_test, y_test) = dataloader(dataset_name)

    # 2. 모델 생성
    model = create_model()

    # 3. 모델 학습
    model = training(model, x_train, y_train, x_test, y_test, epochs=epochs)

    # 4. 학습 완료 후 모델 저장 (선택 사항)
    # model.save('trained_model.h5')
    print("\n모델이 학습이 완료되었습니다.")

    loss, accuracy = model.evaluate(x_test, y_test, verbose=2)

    print(f"\n테스트 데이터 Loss: {loss:.4f}")
    print(f"테스트 데이터 Accuracy: {accuracy:.4f}")



if __name__ == '__main__':
    # argparse로 터미널 인자를 처리합니다.
    parser = argparse.ArgumentParser(description='데이터셋 로드 스크립트')
    parser.add_argument('--dataset_name', nargs="?", default="mnist", type=str, help='로드할 데이터셋의 이름 (예: mnist, fashion_mnist, cifar10)')
    parser.add_argument('--epochs', nargs="?", default=1, type=int, help='훈련 횟수')
    
    args = parser.parse_args()

    # main 함수에 dataset_name을 전달하여 호출
    main(args.dataset_name, args.epochs)
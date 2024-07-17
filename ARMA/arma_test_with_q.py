import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt


def draw_graph(data):
    plt.figure(figsize=(10, 6))
    plt.plot(data['y_t'], label='Actual y_t')
    plt.plot(data['y_t_hat'], label='Predicted y_t_hat', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.title('ARMA(1,1) Model Prediction')
    plt.legend()
    plt.show()


def forecast(data, result):
    # 결과 예측
    forecast = result.predict(start=0, end=len(data) - 1, dynamic=False)
    data['y_t_hat'] = forecast
    print(data)


def fit_arma_model(data):
    # p = 1, q = 1
    model = sm.tsa.ARIMA(data['y_t'], order=(1, 0, 1))
    result = model.fit()
    print(result.summary())
    return result


def open_test_data():
    data = pd.read_csv("arma_data_with_q.csv", delim_whitespace=True, header=None)
    # column data 설정
    data.columns = ['t', 'y_t','epsilon_t_1', 'y_t_1', 'y_t_hat', 'epsilon_t']
    # 0번째 index 제거
    data = data.drop(index=0).reset_index(drop=True)
    # dtype 변환
    data = data.apply(pd.to_numeric)
    print(data.head())
    return data


if __name__ == '__main__':
    data = open_test_data()
    result = fit_arma_model(data)
    forecast(data, result)
    draw_graph(data)

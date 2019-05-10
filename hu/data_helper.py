import numpy as np
from tqdm import tqdm_notebook as tqdm



def gen_data(data_shape):
    lower, upper = -10, 10
    height, width = data_shape
    X_data = np.random.rand(height, width)*(upper - lower) + lower
    print('X_data.shape is', X_data.shape)
    print('X_data[:, 3] is ', X_data[:, 3])

    col1, col2, col3, col4 = 201, 601, 801, 995

    # y_data = (0.3 * X_data[:, col1] ** 2 + 1.1 * X_data[:, col1] * X_data[:, col2] +
    #           0.6 * X_data[:, col2] ** 2 + 0.7 * X_data[:, col2] * X_data[:, col3]) <= 20

    y_data = (0.3*X_data[:,col1]**2 + 2.3*np.sin(X_data[:,col1]*X_data[:,col2]) +
              1.3*X_data[:,col2]**2 + 1.8*np.cos(X_data[:,col1]*X_data[:,col3]) +
              0.8*X_data[:,col4]**3 + 0.7*np.tan(X_data[:,col3]*X_data[:,col4]) +
              0.5*np.exp(X_data[:,col2]) + 0.8*np.tan(X_data[:,col2]*X_data[:,col4]*2.2) +
              0.75*X_data[:,col1]*X_data[:,col2]*X_data[:,col3] -
              0.52*X_data[:,col1]*X_data[:,col3]*X_data[:,col3] +
              0.38*X_data[:,col4]*X_data[:,col2]*X_data[:,col3] -
              0.45*X_data[:,col1]*X_data[:,col2]*X_data[:,col3]*X_data[:,col4]) <= 60

    print('X_data.shape is', X_data.shape)

    y_data_pos = y_data[y_data == 1]
    y_data_neg = y_data[y_data == 0]

    print('y_data_pos.shape y_data_neg.shape is', y_data_pos.shape, y_data_neg.shape)
    print('X_data.shape is', X_data.shape)

    return X_data, y_data

#################################################################################################


#################################################################################################

if __name__ == "__main__":
    X_data, y_data = gen_data((507500, 1000))
    ftrain = open(r'../data/train.txt','w',encoding='utf8')
    ftest = open(r'../data/test.txt','w',encoding='utf8')
    i = 0
    for x,y in tqdm(zip(X_data,y_data),desc='写文件中...'):
        i += 1
        if i % 10000 == 0:
            print('i: ',i // 10000)
        x = list(x)
        x.append(int(y))
        x = [str(_) for _ in x]
        x = ','.join(x)
        if i <= 500000:
            ftrain.write(x+'\n')
        else:
            ftest.write(x+'\n')
    ftrain.close()
    ftest.close()

    


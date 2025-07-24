import numpy as np


def calc_p2p(predict, actual):
    # 计算真正例（True Positive）：预测为正且实际为正的样本数量
    tp = np.sum(predict * actual)
    # 计算真反例（True Negative）：预测为反且实际为反的样本数量
    tn = np.sum((1 - predict) * (1 - actual))
    # 计算假正例（False Positive）：预测为正但实际为反的样本数量
    fp = np.sum(predict * (1 - actual))
    # 计算假反例（False Negative）：预测为反但实际为正的样本数量
    fn = np.sum((1 - predict) * actual)
    # 计算精确率（Precision），为避免除零错误，添加了一个极小值 0.000001
    precision = (tp + 0.000001) / (tp + fp + 0.000001)
    # 计算召回率（Recall），同样为避免除零错误添加了极小值
    recall = (tp + 0.000001) / (tp + fn + 0.000001)
    # 计算 F1 分数，它是精确率和召回率的调和平均数
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall, tp, tn, fp, fn


def point_adjust(score, label, thres):
    # 根据阈值将分数转换为预测结果，大于等于阈值的为正例，否则为反例
    predict = score >= thres
    # 将实际标签转换为布尔类型，大于 0.1 的为正例
    actual = label > 0.1
    # 初始化异常状态为 False
    anomaly_state = False
    for i in range(len(score)):
        # 如果当前样本实际为正例、预测为正例且异常状态为 False
        if actual[i] and predict[i] and not anomaly_state:
            # 将异常状态设为 True
            anomaly_state = True
            for j in range(i, 0, -1):              # 从当前位置向前遍历
                if not actual[j]:                  # 如果遇到实际为反例的样本则停止
                    break
                else:
                    predict[j] = True
        # 如果当前样本实际为反例，将异常状态设为 False
        elif not actual[i]:
            anomaly_state = False
        # 如果处于异常状态，将当前样本的预测结果设为正例
        if anomaly_state:
            predict[i] = True
    return predict, actual


def best_f1_without_pointadjust(score, label):
    # 计算分数的第 99.91 百分位数作为最大阈值
    max_th = np.percentile(score, 99.91)
    # 计算分数的最小值作为最小阈值
    min_th = float(score.min())

    grain = 2000            # 定义阈值搜索的粒度
    max_f1_1 = 0.0          # 初始化最大 F1 分数为 0
    max_f1_th_1 = 0.0       # 初始化最大 F1 分数对应的阈值为 0
    max_pre = 0.0           # 初始化最大精确率为 0
    max_recall = 0.0        # 初始化最大召回率为 0
    for i in range(0, grain + 3):
        # 根据粒度和最大、最小阈值计算当前阈值
        thres = (max_th - min_th) / grain * i + min_th
        actual = label        # 实际标签
        # 根据当前阈值生成预测结果
        predict = score >= thres
        # 调用 calc_p2p 函数计算 F1 分数、精确率、召回率等指标
        f1, precision, recall, tp, tn, fp, fn = calc_p2p(predict, actual)
        if f1 > max_f1_1:           # 如果当前 F1 分数大于最大 F1 分数，更新相关量
            max_f1_1 = f1
            max_f1_th_1 = thres
            max_pre = precision
            max_recall = recall
    # 调用 point_adjust 函数对预测结果进行调整
    predict, actual = point_adjust(score, label, max_f1_th_1)
    return max_f1_1, max_pre, max_recall, predict


def best_f1(score, label):
    # 计算分数的第 99.91 百分位数作为最大阈值
    max_th = np.percentile(score, 99.91)
    # 计算分数的最小值作为最小阈值
    min_th = float(score.min())
    grain = 2000            # 定义阈值搜索的粒度
    max_f1 = 0.0            # 初始化最大 F1 分数为 0
    max_f1_th = 0.0         # 初始化最大 F1 分数对应的阈值为 0
    pre = 0.0               # 初始化精确率为 0
    rec = 0.0               # 初始化召回率为 0
    for i in range(0, grain + 3):
        # 根据粒度和最大、最小阈值计算当前阈值
        thres = (max_th - min_th) / grain * i + min_th
        # 调用 point_adjust 函数对预测结果进行调整
        predict, actual = point_adjust(score, label, thres=thres)
        # 调用 calc_p2p 函数计算 F1 分数、精确率、召回率等指标
        f1, precision, recall, tp, tn, fp, fn = calc_p2p(predict, actual)
        if f1 > max_f1:           # 如果当前 F1 分数大于最大 F1 分数，更新相关量
            max_f1 = f1
            max_f1_th = thres
            pre = precision
            rec = recall
    # 再次调用 point_adjust 函数对预测结果进行调整
    predict, actual = point_adjust(score, label, max_f1_th)
    return max_f1, pre, rec, predict


def get_range_proba(predict, label, delay=7):
    # 找出标签中状态发生变化的位置
    splits = np.where(label[1:] != label[:-1])[0] + 1
    # 判断第一个样本是否为异常样本
    is_anomaly = label[0] == 1
    # 复制预测结果
    new_predict = np.array(predict)
    pos = 0                     # 初始化位置为 0
    for sp in splits:
        if is_anomaly:
            # 如果在当前异常区间内的前 delay+1 个样本中有预测为正例的样本
            if 1 in predict[pos : min(pos + delay + 1, sp)]:
                # 将当前异常区间内的所有样本的预测结果设为正例
                new_predict[pos:sp] = 1
            else:
                # 否则将当前异常区间内的所有样本的预测结果设为反例
                new_predict[pos:sp] = 0
        is_anomaly = not is_anomaly             # 切换异常状态
        pos = sp                                # 更新位置
    # 处理最后一个区间
    sp = len(label)
    if is_anomaly:  # anomaly in the end
        # 如果在最后一个异常区间内的前 delay+1 个样本中有预测为正例的样本
        if 1 in predict[pos : min(pos + delay + 1, sp)]:
            # 将最后一个异常区间内的所有样本的预测结果设为正例
            new_predict[pos:sp] = 1
        else:
            # 否则将最后一个异常区间内的所有样本的预测结果设为反例
            new_predict[pos:sp] = 0
    return new_predict


def delay_f1(score, label, k=7):
    # 计算分数的第 99.91 百分位数作为最大阈值
    max_th = np.percentile(score, 99.91)
    # 计算分数的最小值作为最小阈值
    min_th = float(score.min())
    grain = 2000                # 定义阈值搜索的粒度
    max_f1 = 0.0                # 初始化最大 F1 分数为 0
    max_f1_th = 0.0             # 初始化最大 F1 分数对应的阈值为 0
    pre = 0.0                   # 初始化精确率为 0
    rec = 0.0                   # 初始化召回率为 0
    for i in range(0, grain + 3):
        # 根据粒度和最大、最小阈值计算当前阈值
        thres = (max_th - min_th) / grain * i + min_th
        # 根据当前阈值生成预测结果
        predict = score >= thres
        # 调用 get_range_proba 函数对预测结果进行调整
        predict = get_range_proba(predict, label, k)
        # 调用 calc_p2p 函数计算 F1 分数、精确率、召回率等指标
        f1, precision, recall, tp, tn, fp, fn = calc_p2p(predict, label)
        if f1 > max_f1:           # 如果当前 F1 分数大于最大 F1 分数，更新相关量
            max_f1 = f1
            max_f1_th = thres
            pre = precision
            rec = recall
        # 再次调用 get_range_proba 函数对预测结果进行调整
        predict = get_range_proba(score >= max_f1_th, label, k)
    return max_f1, pre, rec, predict
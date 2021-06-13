# coding: utf-8
# 2021/3/28 @ liujiayu

import logging
import numpy as np
from tqdm import tqdm
import pickle
from EduCDM import CDM
from itertools import product


def initial_all_knowledge_state(know_num):
    state_num = 2 ** know_num
    all_states = np.zeros((state_num, know_num))
    for i in range(state_num):
        k, quotient, residue = 1, i // 2, i % 2  # quotient:商 residue:余数
        while True:
            all_states[i, know_num - k] = residue
            if quotient <= 0:
                break
            quotient, residue = quotient // 2, quotient % 2
            k += 1
    return all_states

def _get_all_skills(know_num):
    # 获得所有可能被试技能的排列组合
    size = know_num
    # 取【0，1】的笛卡尔积，repeat指定iterable重复几次
    return np.array(list(product([0, 1], repeat=size)))


def init_parameters(stu_num, prob_num):
    slip = np.zeros(shape=prob_num) + 0.2
    guess = np.zeros(shape=prob_num) + 0.2
    theta = np.zeros(shape=stu_num)  # index of state
    return theta, slip, guess


class DINA(CDM):
    """
        DINA model, training (EM) and testing methods
        :param R (array): response matrix, shape = (stu_num, prob_num)
        :param q_m (array): Q matrix, shape = (prob_num, know_num)
        :param stu_num (int): number of students
        :param prob_num (int): number of problems
        :param know_num (int): number of knowledge
        :param skip_value (int): skip value in response matrix
    """

    def __init__(self, R, q_m, stu_num, prob_num, know_num, skip_value=-1):
        self.R, self.q_m, self.state_num, self.skip_value = R, q_m, 2 ** know_num, skip_value
        self.stu_num, self.prob_num, self.know_num = stu_num, prob_num, know_num
        self.theta, self.slip, self.guess = init_parameters(stu_num, prob_num)
        self.all_states = initial_all_knowledge_state(know_num)  # shape = (state_num, know_num)
        # keepdims:保持其二维性
        # np.sum(q, axis=1) sum each row. Size = question * 1
        # Transpose-> 1 * question
        # 1 * question - question * (2**k) = (2**k) * question
        # state_prob里大于0的元素表示无法答对该试题，具体数字表示缺少的掌握知识点个数。
        state_prob = np.transpose(np.sum(q_m, axis=1, keepdims=True) - np.dot(q_m, np.transpose(self.all_states)))
        self.eta = 1 - (state_prob > 0)  # state covers knowledge of problem (1: yes), shape = (state_num, prob_num)

    def train(self, epoch, epsilon) -> ...:
        like = np.zeros(shape=(self.stu_num, self.state_num))  # likelihood
        post = np.zeros(shape=(self.stu_num, self.state_num))  # posterior 后验
        theta, slip, guess, tmp_R = np.copy(self.theta), np.copy(self.slip), np.copy(self.guess), np.copy(self.R)
        tmp_R[np.where(self.R == self.skip_value)[0], np.where(self.R == self.skip_value)[1]] = 0
        for iteration in range(epoch):
            post_tmp, slip_tmp, guess_tmp = np.copy(post), np.copy(slip), np.copy(guess)
            answer_right = (1 - slip) * self.eta + guess * (1 - self.eta)  # (2**k) * question
            for s in range(self.state_num):  # O(2**k) Loop
                # question * (stu * question) = stu * question
                log_like = np.log(answer_right[s, :] + 1e-9) * self.R + np.log(1 - answer_right[s, :] + 1e-9) * (
                    1 - self.R)
                # except the unknown data
                log_like[np.where(self.R == self.skip_value)[0], np.where(self.R == self.skip_value)[1]] = 0
                # sum each row. Shape = stu_num
                like[:, s] = np.exp(np.sum(log_like, axis=1))
            # compute the posterior. Like's shape = stu * (2**k)
            # 这里假设所有认知状态的存在概率是相同的，即先验概率相同，因此计算后验时可以约掉所有的先验概率，
            post = like / np.sum(like, axis=1, keepdims=True)

            # np.sum(post, axis=0) sum each column. Shape = 2 ** k
            # post's Shape = (2**k, None)
            # np.expand_dims(post, axis=1)表示在1位置添加数据
            # (2 ** k, None) expand -> (2**k) * 1
            i_l = np.expand_dims(np.sum(post, axis=0), axis=1)  # shape = (state_num, 1)

            # ((2**k) * stu) dot (stu * question) = (2**k) * question
            # sum(每个认知状态对每个学生对应的后验概率 * 每个学生在同一试题上的得分) = r_jl(j=question,l=cognitive status)
            r_jl = np.dot(np.transpose(post), tmp_R)  # shape = (state_num, prob_num)

            # eta's shape = (2**k) * question
            # ((2**k) * question) * ((2**k) * question) = ((2**k) * question)
            # sum(axis=0) -> shape = question_num
            r_jl_0, r_jl_1 = np.sum(r_jl * (1 - self.eta), axis=0), np.sum(r_jl * self.eta, axis=0)
            i_jl_0, i_jl_1 = np.sum(i_l * (1 - self.eta), axis=0), np.sum(i_l * self.eta, axis=0)
            guess, slip = r_jl_0 / i_jl_0, (i_jl_1 - r_jl_1) / i_jl_1

            change = max(np.max(np.abs(post - post_tmp)), np.max(np.abs(slip - slip_tmp)),
                         np.max(np.abs(guess - guess_tmp)))
            # shape = stu_num, find student's cognitive status
            theta = np.argmax(post, axis=1)
            if iteration > 20 and change < epsilon:
                break
        self.theta, self.slip, self.guess = theta, slip, guess

    def eval(self, test_data) -> tuple:
        pred_score = (1 - self.slip) * self.eta + self.guess * (1 - self.eta)
        test_rmse, test_mae = [], []
        for i in tqdm(test_data, "evaluating"):
            stu, test_id, true_score = i['user_id'], i['item_id'], i['score']
            test_rmse.append((pred_score[self.theta[stu], test_id] - true_score) ** 2)
            test_mae.append(abs(pred_score[self.theta[stu], test_id] - true_score))
        return np.sqrt(np.average(test_rmse)), np.average(test_mae)

    def save(self, filepath):
        with open(filepath, 'wb') as file:
            pickle.dump({"theta": self.theta, "slip": self.slip, "guess": self.guess}, file)
            logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        with open(filepath, 'rb') as file:
            self.theta, self.slip, self.guess = pickle.load(file).values()
            logging.info("load parameters from %s" % filepath)

    def inc_train(self, inc_train_data, epoch, epsilon):  # incremental training
        for i in inc_train_data:
            stu, test_id, true_score = i['user_id'], i['item_id'], i['score']
            self.R[stu, test_id] = true_score
        self.train(epoch, epsilon)

    def transform(self, records):  # MLE for evaluating student's state
        # max_like_id: diagnose which state among all_states the student belongs to
        # dia_state: binaray vector of length know_num, 0/1 indicates whether masters the knowledge
        answer_right = (1 - self.slip) * self.eta + self.guess * (1 - self.eta)
        log_like = records * np.log(answer_right + 1e-9) + (1 - records) * np.log(1 - answer_right + 1e-9)
        log_like[:, np.where(records == self.skip_value)[0]] = 0
        max_like_id = np.argmax(np.exp(np.sum(log_like, axis=1)))
        dia_state = self.all_states[max_like_id]
        return max_like_id, dia_state


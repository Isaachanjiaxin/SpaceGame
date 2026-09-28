from numpy import ndarray, kron, ones_like, ones
import numpy as np
from warnings import warn
from typing import Optional
# 定义物理常量
G = 6.674e-11       # 引力常量
ROU = 1e10           # 系数常量 用于默认计算星球的质量


# 定义物理规则
def Gravity(Ms:Optional[ndarray]=None,
            pos:Optional[ndarray]=None,
            self_pos:Optional[ndarray]=None,
            debug:bool=False) -> Optional[ndarray]:
    """
    :param Ms: ndim = 3     用于记录除自己外每个天体的质量
    :param pos: ndim = 3    用于记录除自己外所有天体的位置
    :param self_pos: ndim = 3    用于记录每个天体自己的位置
    :return: n个列向量，对各个self_pos的引力加速度


    数据结构
                            Ms       pos      self_pos
    shape[0](天体个数)        n        n        n
    shape[1](维度)           1        3        3
    shape[2](位置个数)        n-1      n-1      1
    """
    if Ms is None or pos is None or self_pos is None:
        return None
    # 调整Ms的维度为3
    if Ms.ndim != 3:
        warn("Ms must have 3 tensor dimension")
        Ms = Ms.reshape((Ms.shape[0],1,Ms.shape[1]))

    if Ms.shape[0]!= pos.shape[0]:
        raise ValueError(f"Ms and pos must have the same length {Ms.shape[0]} != {pos.shape[0]}")
    if pos.shape[1]!= self_pos.shape[1]:
        raise ValueError(f"pos and self_pos must have the same dimension {pos.shape[1]} != {self_pos.shape[0]}")
    if self_pos.shape[2] != 1:
        raise ValueError(f"self_pos must be a column vector {self_pos.shape[2]} != 1")
    if Ms.shape[1] != 1:
        raise ValueError(f"Ms only have one dimension {Ms.shape[1]} != 1")

    # 先保留self_pos的形状，方便后续reshape
    res_shape = self_pos.shape

    # 扩展self_pos，方便后续计算距离
    self_pos = kron(self_pos,ones((1,1,pos.shape[2])))
    delts_pos = pos - self_pos
    if debug:
        print("delts_pos:\n",delts_pos)
    Rs = np.linalg.norm(delts_pos, axis=1).reshape(*Ms.shape)
    if debug:
        print("Rs:\n",Rs)
    tmp_acc = (G * Ms / (Rs**3)) * delts_pos
    if debug:
        print("Ms:\n",Ms)
        print("Rs**3:\n",Rs**3)
        print("G * Ms / Rs**3:\n",(G * Ms / Rs**3))
        print("delts_pos:\n",delts_pos)
        print("tmp_acc:\n",tmp_acc)
    acc = tmp_acc.sum(axis=2).reshape(res_shape)
    return acc

def _test_Gravity(m1:float,m2:float,pos1:ndarray,pos2:ndarray) -> tuple[ndarray,ndarray]:
    """用于检查引力加速度是否正确"""
    acc1 = G*m2/np.linalg.norm(pos2-pos1)**3 * (pos2-pos1)
    acc2 = G*m1/np.linalg.norm(pos2-pos1)**3 * (pos1-pos2)
    return acc1,acc2
import numpy as np
from numpy import array, ndarray, pi
from typing import Tuple,List,Union,Optional
from Rule import ROU, Gravity
"""约定：
1. 所有物品的坐标都必须是3维的
2. 所有位置信息都是列向量
"""
MORANDI_COLORS = [
    (0.678, 0.596, 0.510),
    (0.549, 0.573, 0.471),
    (0.635, 0.522, 0.471),
    (0.482, 0.522, 0.506),
    (0.722, 0.604, 0.561),
    (0.604, 0.553, 0.510),
    (0.451, 0.471, 0.494),
    (0.690, 0.635, 0.592),
    (0.522, 0.514, 0.471),
    (0.584, 0.612, 0.569),
    (0.773, 0.722, 0.643),
    (0.502, 0.510, 0.541),
    (0.682, 0.639, 0.565),
    (0.608, 0.573, 0.533),
    (0.545, 0.545, 0.522),
]

class Star:
    def __init__(
            self,
            pos:ndarray=array([0,0,0]).reshape(-1,1),
            radius:int=5,
            color:Optional[Union[Tuple,str]]=None,
            vel:Optional[ndarray]=None,
            mass:Optional[float]=None,
    ):
        self.pos:ndarray = array(pos,dtype=np.float64).reshape(-1,1)
        self.radius:float = radius
        self.V:float = (4*pi/3)*self.radius**3
        self.mass:float = mass if mass is not None else (4*pi/3)*self.radius**3 * ROU
        self.vel:ndarray = array(vel,dtype=np.float64).reshape(-1,1) if vel is not None\
            else array([0,0,0],dtype=np.float64).reshape(-1,1)
        if isinstance(color,Tuple):
            self.color = color
        elif isinstance(color,str):
            if color == "red":
                self.color = (1.0, 0.0, 0.0)
            elif color == "green":
                self.color = (0.0, 1.0, 0.0)
            elif color == "blue":
                self.color = (0.0, 0.0, 1.0)
            else:
                raise ValueError("color must be red, green, or blue")
        else:
            idx = np.random.randint(0,len(MORANDI_COLORS))
            self.color = MORANDI_COLORS[idx]

    def __str__(self):
        return f"""Star(pos={self.pos.flatten()}, radius={self.radius}, color={self.color}, vel={self.vel.flatten()},  mass={self.mass})"""

    def __add__(self, other):
        if not isinstance(other, Star):
            raise TypeError(f"Star can only be added to other Star, but got {type(other)}")
        new_star = Star(
            pos=(self.pos + other.pos)/2,
            mass=self.mass + other.mass,
        )
        new_star.V = self.V + other.V
        new_star.radius = (3*(new_star.V)/(4*pi))**(1/3)
        new_star.vel = (self.vel * self.mass + other.vel * other.mass)/(self.mass + other.mass)
        return new_star

def generate_hub(M:Optional[ndarray]=None,pos:Optional[ndarray]=None)->Tuple[Optional[ndarray],Optional[ndarray]]:
    """生成一个星系计算需要的张量"""
    if M is None or pos is None:
        return None, None
    M_hub,pos_hub = None,None
    if M.shape[-1] != pos.shape[0]:
        raise ValueError(f"M and pos must have same size, but M.shape={M.shape} and pos.shape={pos.shape}")
    if M.ndim != 2:
        M = M.reshape(1, -1)
    M_hub, pos_hub = None, None
    for idx in range(M.shape[-1]):
        # 生成M_hub
        if M_hub is None:
            M_hub = np.hstack((M[:,:idx],M[:,idx+1:])).reshape(1, 1, -1)
        else:
            tmp = np.hstack((M[:,:idx],M[:,idx+1:])).reshape(1, 1, -1)
            M_hub = np.concatenate((M_hub,tmp),axis=0)
        # 生成pos_hub
        if pos_hub is None:
            tmp1 = np.concatenate((pos[:idx,:,:],pos[idx+1:,:,:]),axis=0)
            tmp2 = np.concatenate(tmp1,axis=1)
            pos_hub = tmp2.reshape(1,*tmp2.shape)
        else:
            tmp1 = np.concatenate((pos[:idx,:,:],pos[idx+1:,:,:]),axis=0)
            tmp2 = np.concatenate(tmp1,axis=1)
            tmp3 = tmp2.reshape(1,*tmp2.shape)
            pos_hub = np.concatenate((pos_hub,tmp3),axis=0)
    return M_hub,pos_hub

def convert_to_plot(vec:Optional[ndarray]=None)->Optional[ndarray]:
    return np.concatenate(vec,axis=1) if vec is not None else None

class System:
    def __init__(
            self,
            stars:List[Star],
    ):
        self.stars = stars
        M,pos = None,None
        radius,vel = None,None
        colors = []
        for star in self.stars:
            # M [[M1,M2,M3,...]]
            if M is None:
                M = [star.mass]
            else:
                M.append(star.mass)
            # pos [[pos1],[pos2],[pos3],...]] 每个元素是一个列向量
            if pos is None:
                pos = star.pos.reshape(1,*star.pos.shape)
            else:
                pos = np.concatenate((pos,star.pos.reshape(1,*star.pos.shape)),axis=0)
            if vel is None:
                vel = star.vel.reshape(1,*star.vel.shape)
            else:
                vel = np.concatenate((vel,star.vel.reshape(1,*star.vel.shape)),axis=0)
            if radius is None:
                radius = [star.radius]
            else:
                radius.append(star.radius)

            colors.append(star.color)
        # 一个星系的基本参数
        self.M = np.array(M).reshape(1,-1)
        self.pos = pos
        self.M_hub, self.pos_hub = generate_hub(self.M, self.pos)
        self.radius = array(radius)
        self.vel = vel
        self.colors = colors
        # 用于绘制的pos
        self.plot_pos = convert_to_plot(self.pos)

    def __str__(self):
        return f"""System(\nM:\n{self.M.flatten()},\npos:\n{self.pos}\n)"""

    def __repr__(self) -> str:
        return f"""System(\nM:\n{self.M.flatten()},\npos:\n{self.pos}\nradius:\n{self.radius}\nvel:\n{self.vel}\n)"""

    def step(self, dt:float):
        """更新系统中的所有星体的位置和速度"""
        acc = Gravity(self.M_hub, self.pos_hub, self.pos)
        self.pos += self.vel*dt
        self.vel += acc*dt
        self.M_hub, self.pos_hub = generate_hub(self.M, self.pos)
        self.plot_pos = convert_to_plot(self.pos)

if __name__ == "__main__":
    star1 = Star(
        pos=array([0,0,0]),
        radius=3,
        color=(0,1,0),
        vel=array([0,0,0]),
    )
    star2 = Star(
        pos=array([1,0,0]),
        radius=3,
        color=(0,1,0),
        vel=array([0,0,0]),
    )
    star3 = Star(
        pos=array([0,1,0]),
        radius=3,
        color=(0,1,0),
        vel=array([0,0,0]),
    )
    system = System(
        stars=[star1, star2, star3],
    )
    print(system)
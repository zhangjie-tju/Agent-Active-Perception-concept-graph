import numpy as np

def bbox_to_corners(center, extent):
    """
    将边界框中心和尺寸转换为边界框的角点坐标。
    """
    min_corner = center - extent / 2
    max_corner = center + extent / 2
    return min_corner, max_corner

def is_inside(a_min, a_max, b_min, b_max):
    """
    判断物体a是否完全在物体b的边界框内。
    """
    return (a_min >= b_min).all() and (a_max <= b_max).all()

def is_above(a_max,a_min,b_max,b_min):
    """
    判断物体a是否在物体b的上面。
    """
    return a_min[2] >= b_min[2] and a_max[2]>=b_max[2]

def is_below(a_min, b_max):
    """
    判断物体a是否在物体b的下面。
    """
    return a_min[1] <= b_max[1]

def analyze_position(a_center, a_extent, b_center, b_extent):
    """
    分析两个物体的位置关系。
    """
    # 计算物体a和b的最小和最大坐标
    a_min, a_max = bbox_to_corners(a_center, a_extent)
    b_min, b_max = bbox_to_corners(b_center, b_extent)

    # 判断a是否在b里面
    a_in_b = is_inside(a_min, a_max, b_min, b_max)
    # 判断b是否在a里面
    b_in_a = is_inside(b_min, b_max, a_min, a_max)

    if a_in_b:
        return "a in b"
    elif b_in_a:
        return "b in a"
    else:
        # 判断a是否在b上面
        a_above_b = is_above(a_max,a_min,b_max, b_min)
        # 判断b是否在a上面
        b_above_a = is_above(b_max, b_min,a_max, a_min)
        
        if a_above_b and not b_above_a:
            return "a on b"
        elif b_above_a and not a_above_b:
            return "b on a"
        else:
            return "none of these"
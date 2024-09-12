import numpy as np

def normal_mach_number(u, normal_vector, ga):
    velocity = np.array([u[1], u[2]]) / u[0]
    magnitude_of_v = np.linalg.norm(velocity)
    normal_velocity = velocity[0] * normal_vector[0] + velocity[1] * normal_vector[1]

    p = (ga - 1) * (u[3] - 0.5 * u[0] * magnitude_of_v**2)
    c = np.sqrt(ga * p / u[0])

    m_normal = normal_velocity / c
    return m_normal

import numpy as np
import random
import math

# ===============
# Parameters

params_SocialForceYang = {
    # Constraint Parameters for Speed and Acceleration
    # statistical values
    'v_max': 2.5,
    'v_normal': 1.7,
    'v_dense': 0.3,
    'a_max': 5,
    'a_normal': 2.5,
    'a_dense': 0.68,
    # sparseness to speed
    'beta_v_S': 3.9761,
    'S_v_0': 0.06566917,
    # vehicle to speed
    'beta_v_F': 0.001577598,
    'F_v_0': 199.3611,
    # sparseness to acc
    'beta_a_S': 2.994062,
    'S_a_0': 0.39941,
    # vehicle to acc
    'beta_a_F': 0.09775474,
    'F_a_0': 53.94855,

    # Pedestrian-Pedestrian Interaction
    # collision
    'col_alpha': 9825.125,
    # Repulsion
    'rep_d0': 0.7801,
    'rep_F0': 301.028,
    'rep_sigma': 0.45971243,
    'rep_ani_lambda': 0.1,
    # Navigation
    'nav_d0': 1.5892008,  # problematic ???
    'nav_F0': 410.875,
    'nav_sigma': 0.41745,
    'nav_ani_lambda': 1,
    # Sparseness
    'sparseness_thr': 3.665375,  #
    'sparseness_fov': 121.39191,  # unit: angle, problematic ???
    'sparseness_lambda': 1.87,

    # Vehicle-Pedestrian Interaction
    'le': 0.2151011,
    'veh_d_x_0': 0.510985,
    'veh_alpha_x': 1.394358,
    'veh_A': 777.5852,
    'veh_b': 2.613755,
    'veh_ani_lambda': 0.3119132,

    # Destination Driven
    'des_vd': 1.394293,  # DP
    'des_sigma': 0.3,  # DP
    'des_k': 545.3125,  # non - VIP
    'F_1': 199.7455,
    'F_2': 672.6487
}

params_category = {
    'constraints': [
        # statistical values
        'v_max',
        'v_max',
        'v_normal',
        'a_max',
        'a_normal',
        'a_dense',
        # sparseness to speed
        'beta_v_S',
        'S_v_0',
        # vehicle to speed
        'beta_v_F',
        'F_v_0',
        # sparseness to acc
        'beta_a_S',
        'S_a_0',
        # vehicle to acc
        'beta_a_F',
        'F_a_0'
    ],
    'p2p_interaction': [
        # collision
        'col_alpha',
        # Repulsion
        'rep_d0',
        'rep_F0',
        'rep_sigma',
        'rep_ani_lambda',
        # Navigation
        'nav_d0',
        'nav_F0',
        'nav_sigma',
        'nav_ani_lambda',
        # Sparseness
        'sparseness_thr',
        'sparseness_fov',
        'sparseness_lambda'
    ],
    'v2p_interaction': [
        'le',
        'veh_d_x_0',
        'veh_alpha_x',
        'veh_A',
        'veh_b',
        'veh_ani_lambda',
    ],
    'destination': [
        'des_vd',
        'des_sigma',
        'des_k',
        'F_1',
        'F_2'
    ]
}


def print_params(params: dict, categories=None):
    if categories is None:
        for param, value in params.items():
            print(f'{param:<20} = {value:>15.4f}')
    else:
        for cat_name, cat_values in categories.items():
            print(f'===> parameters for {cat_name}')
            for key in cat_values:
                print(f'{key:<20} = {params[key]:>15.4f}')
    print('=============================================')
    return None


# ========================================================
# Motion Model: Social Force - Old Version (function only)
# TODO @ modify social force model so that it records the history trajectory of the calculated force.


def calculate_total_force(params, ego, surs, vehs, env, des, dt):
    """

    :param params: a dictionary of parameters of the social force model
    :param ego: state of ego pedestrian, a numpy array of shape (4,), [x, y, vx, vy]
    :param surs: states of surrounding ped. a numpy array of shape (n, 4).
    'n' is the number of surrounding peds, '4' represents state [x, y, vx, vy].
    :param vehs: states of surrounding modules_vehicle. a numpy array of shape (n, 4).
    'n' is the number of surrounding peds, '4' represents state [x, y, yaw, speed].
    :param env: empty for now, you define it
    :param des: position of the temporal goal, a numpy array of shape (2,), [x, y]
    :param dt: time step
    :return: the force vector applied on the ego pedestrian
    """
    # todo @ check before passing params into the function, check if there are repeated key values.
    des = np.array(des).reshape(2, 1)
    ego = np.array(ego).reshape(4, 1)
    f_rep_total, f_col_total, f_nav_total, spa, spa_count = __force_p2p(params, ego, surs)
    f_veh_total, _, _ = __force_v2p(params, ego, vehs)
    f_des = __force_d2p(params, ego, des, f_veh_total)
    f_env = __force_e2p(params)

    f_total = f_rep_total + f_col_total + f_nav_total + f_veh_total + f_des + f_env
    f_total_adjusted = __force_adjust(params=params, F=f_total, F_veh=f_veh_total, spa=spa, v_now=ego[2:], dt=dt)

    # return f_total_adjusted # todo @ need change scripts that call this function
    return f_total_adjusted, f_rep_total, f_col_total, f_nav_total, f_des, f_veh_total


def __force_p2p(params, ego, surs):
    sparseness_peds_count = 0
    sparseness = 1000
    # f_ped_total = np.array([[0], [0]])
    f_col_total = np.array([[0], [0]])
    f_rep_total = np.array([[0], [0]])
    f_nav_total = np.array([[0], [0]])

    for j, sur in enumerate(surs):

        if math.isnan(sur[0]) or math.isnan(sur[1]):
            continue

        sur = sur.reshape(4, 1)

        # spatial relationship
        r_ij = sur[:2] - ego[:2]
        v_ij = sur[2:] - ego[2:]
        dc_ij = np.linalg.norm(r_ij)
        db_ij = dc_ij - 2 * params['R']
        n_ij = r_ij / dc_ij
        t_ij = np.array([n_ij[1], -n_ij[0]])  # right side direction

        # angle between v_i (walking direction) and n_ij
        if all(ego[2:] == 0):
            cos_phi_v_i_n_ij = 0
        else:
            cos_phi_v_i_n_ij = np.dot(r_ij.transpose(), ego[2:]) / (np.linalg.norm(r_ij) * np.linalg.norm(ego[2:]))

        # angle between v_ji (relative velocity, i walking to j, in the coord. of j) and n_ij
        if all(v_ij == 0):
            cos_phi_v_ji_n_ij = 0
        else:
            cos_phi_v_ji_n_ij = np.dot(-v_ij.transpose(), n_ij) / (np.linalg.norm(-v_ij) * np.linalg.norm(n_ij))

        # Collision
        if db_ij < 0:
            col_mag, _, _ = F_magnitude_collision(db_ij, params['col_alpha'])
        else:
            col_mag = 0
        F_col = - col_mag * n_ij

        # Repulsion
        rep_mag, _, _ = F_magnitude_repulsion(db_ij, params['rep_d0'], params['rep_F0'], params['rep_sigma'],
                                              cos_phi_v_i_n_ij, params['rep_ani_lambda'])
        F_rep = - rep_mag * n_ij

        # Navigation
        nav_sign = np.sign(np.dot(-v_ij.transpose(), t_ij))
        if nav_sign == 0:
            if random.randint(0, 1):
                nav_sign = 1
            else:
                nav_sign = -1
        nav_mag, _, _ = F_magnitude_navigation(db_ij, params['nav_d0'], params['nav_F0'], params['nav_sigma'],
                                               cos_phi_v_ji_n_ij, params['nav_ani_lambda'])
        F_nav = nav_mag * nav_sign * t_ij

        # Calculating density information
        if cos_phi_v_i_n_ij > math.cos(
                params['sparseness_fov'] / 2 * np.pi / 180) and db_ij < params['sparseness_thr']:
            sparseness_peds_count = sparseness_peds_count + 1
            sparseness_j = fun_sparseness_legacy(db_ij, cos_phi_v_i_n_ij, params['sparseness_lambda'])
            # sparseness_j = db_ij / fun_anisotropy_linear(cos_phi_v_i_n_ij, params['sparseness_lambda'])
            if sparseness_j < sparseness:
                sparseness = sparseness_j

        # update total forces
        # f_ped_total = f_ped_total + F_col + F_rep + F_nav # Total force from ped j
        f_col_total = f_col_total + F_col
        f_nav_total = f_nav_total + F_nav
        f_rep_total = f_rep_total + F_rep

    # f_ped_total = f_col_total + f_nav_total + f_rep_total

    return f_rep_total, f_col_total, f_nav_total, sparseness, sparseness_peds_count


def __force_v2p(params, ego, vehs):
    f_vehs = np.array([[0], [0]])
    outside = 1
    vec_n = np.array([[0], [0]])

    for v, veh in enumerate(vehs):

        if math.isnan(veh[0]) or math.isnan(veh[1]):
            continue

        # print('test')

        veh = veh.reshape(4, 1)
        veh_theta = veh[2]
        veh_u = veh[3]

        # spatial relationship
        vec_v2p = ego[:2] - veh[:2]
        dv2p = np.linalg.norm(vec_v2p)
        theta_v2p = math.atan2(vec_v2p[1], vec_v2p[0])
        theta_v2p_local = correct_angle_range(theta_v2p - veh_theta)
        ped_x_local = dv2p * math.cos(theta_v2p_local)
        ped_y_local = dv2p * math.sin(theta_v2p_local)
        ped_xy_local = np.array([ped_x_local, ped_y_local])
        ped_xy_local = np.expand_dims(ped_xy_local, axis=1)

        ped_v_xy_local = np.array([[math.cos(veh_theta), math.sin(veh_theta)],
                                   [-math.sin(veh_theta), math.cos(veh_theta)]]).dot(ego[2:])

        # generating virtual shape
        ped_R = params['R']
        d_x = params['veh_alpha_x'] * veh_u
        lf = params['LF']
        lr = params['LR']
        lw = params['WIDTH']
        le = params['le']
        veh_d_x_0 = params['veh_d_x_0']

        outside = 1
        veh_x_source = 0
        veh_y_source = 0

        # the most front point
        B = np.expand_dims(np.array(
            [float(lf + le + veh_d_x_0 + d_x), 0]
        ), axis=1)

        A_neg = np.expand_dims(np.array(
            [float(lf + le + veh_d_x_0), -(lw / 2 + le)]
        ), axis=1)

        A_pos = np.expand_dims(np.array(
            [float(lf + le + veh_d_x_0), lw / 2 + le]
        ), axis=1)

        # find the closest point on the virtual shape to the pedestrian

        # behind the vehicle
        if ped_x_local < -(lr + le):
            if ped_y_local < -(lw / 2 + le):
                veh_x_source = -(lr + le)
                veh_y_source = -(lw / 2 + le)
            elif ped_y_local >= -(lw / 2 + le) and ped_y_local < (lw / 2 + le):
                veh_x_source = -(lr + le)
                veh_y_source = ped_y_local
            elif ped_y_local >= (lw / 2 + le):
                veh_x_source = -(lr + le)
                veh_y_source = (lw / 2 + le)
            veh_xy_source = np.expand_dims(np.array([veh_x_source, veh_y_source]), axis=1)

        # between extended front bumper and rear bumper
        elif ped_x_local >= -(lr + le) and ped_x_local < (lf + le + veh_d_x_0):
            if ped_y_local < 0:
                veh_x_source = ped_x_local
                veh_y_source = -(lw / 2 + le)
            elif ped_y_local >= 0:
                veh_x_source = ped_x_local
                veh_y_source = (lw / 2 + le)
            veh_xy_source = np.expand_dims(np.array([veh_x_source, veh_y_source]), axis=1)
            # check if ped inside the inflential area
            if ped_y_local > -(lw / 2 + le) and ped_y_local < (lw / 2 + le):
                outside = 0

        # in front of extended front bumper
        elif ped_x_local >= (lf + le + veh_d_x_0):
            vertices = [(A_neg, B), (A_pos, B)]
            veh_xy_source = closest_point_on_line_segments(ped_xy_local, vertices)
            if ped_x_local < (lf + le + veh_d_x_0 + d_x) and \
                    abs(ped_y_local) < (le + lw / 2) - (le + lw / 2) / d_x * (ped_x_local - (lf + le + veh_d_x_0)):
                outside = 0

            # k = d_x / (lw / 2 + le)
            # if (ped_x_local - (lf + le + veh_d_x_0)) < d_x - k * abs(ped_y_local) and ped_x_local < (lf+le+veh_d_x_0+d_x):
            #     outside = 0

        # calculate force by using anitrosopy
        vec_dist = ped_xy_local - veh_xy_source
        dist = np.linalg.norm(vec_dist)
        vec_n = vec_dist / dist
        if not outside:
            dist = -dist
            vec_n = -vec_n

        # cos_theta_veh_xy_source = np.dot(-vec_n.transpose(), ped_v_xy_local) / (np.linalg.norm(vec_n) * np.linalg.norm(ped_v_xy_local))

        v_p2v_local = ped_v_xy_local - np.array([list(veh_u), [0]])
        if all(v_p2v_local == 0):
            v_p2v_local = np.array([[-0.01], [0]])
        cos_theta_veh_xy_source = np.dot(-vec_v2p.transpose(), v_p2v_local) / (
                np.linalg.norm(vec_v2p) * np.linalg.norm(v_p2v_local))

        effect_longi = 1

        # force

        veh_mag = fun_decaying_exp(dist - ped_R, params['veh_A'], params['veh_b'])
        veh_ani = fun_anisotropy_sinusoid(cos_theta_veh_xy_source, params['veh_ani_lambda'])

        if dist - ped_R < 0:
            F_veh_local = veh_mag * effect_longi * vec_n
        else:
            F_veh_local = veh_mag * veh_ani * effect_longi * vec_n

        # conver back
        f_veh = np.array([[math.cos(veh_theta), -math.sin(veh_theta)], [math.sin(veh_theta), math.cos(veh_theta)]]).dot(
            F_veh_local)
        f_vehs = f_vehs + f_veh

    # print('test')

    return f_vehs, outside, vec_n


def __force_d2p(params, ego, des, f_vehs):
    des = des.reshape(2, 1)

    vec_vd = fun_des_vd(ego[:2], des, params['des_vd'], params['des_sigma'])
    beta = fun_des_beta(f_vehs, params['F_1'], params['F_2'])

    f_des = beta * params['des_k'] * (vec_vd - ego[2:])

    return f_des


def __force_e2p(params):
    # TODO @ future work

    return np.array([[0], [0]])


def __force_adjust(params, F, F_veh, spa, v_now, dt):
    # obtain constraints
    v_limit, _, _ = fun_lim_v(spa,
                              np.linalg.norm(F_veh),
                              params['beta_v_S'],
                              params['S_v_0'],
                              params['beta_v_F'],
                              params['F_v_0'],
                              params['v_max'],
                              params['v_normal'],
                              params['v_dense'])
    a_limit, _, _ = fun_lim_a(spa,
                              np.linalg.norm(F_veh),
                              params['beta_a_S'],
                              params['S_a_0'],
                              params['beta_a_F'],
                              params['F_a_0'],
                              params['a_max'],
                              params['a_normal'],
                              params['a_dense'])

    ped_m = params['m']
    # acceleration
    if np.linalg.norm(F) > ped_m * a_limit:
        F = F / np.linalg.norm(F) * ped_m * a_limit

    # speed todo check the type of v_now
    v_new = v_now + dt * F / ped_m
    if np.linalg.norm(v_new) > v_limit:
        F = ped_m / dt * (v_new / np.linalg.norm(v_new) * v_limit - v_now)

    return F


# ===============
# Basic Functions


def F_magnitude_collision(d, alpha):
    mag = -alpha * min(d, 0)
    ani = 1
    return mag * ani, mag, ani


def F_magnitude_repulsion(d, d0, F0, sigma, cos_phi, la):
    mag = fun_decaying_lm(d, d0, F0, sigma)
    ani = fun_anisotropy_sinusoid(cos_phi, la)
    return mag * ani, mag, ani


def F_magnitude_navigation(d, d0, F0, sigma, cos_phi, la):
    mag = fun_decaying_lm(d, d0, F0, sigma)
    ani = fun_anisotropy_exp(cos_phi, la)
    return mag * ani, mag, ani


def fun_decaying_lm(d, d0, M, sigma):
    if sigma == 0:
        m = M / d0 * max(0.0, d0 - d)
    else:
        m = M / (2 * d0) * (d0 - d + np.sqrt((d0 - d) ** 2 + sigma ** 2))
    return m


def fun_decaying_exp(d, A, b):
    return A * np.exp(- b * d)


def fun_anisotropy_sinusoid(cos_phi, Lambda):
    return Lambda+(1-Lambda)*(1+cos_phi)/2


def fun_anisotropy_exp(cos_phi, Lambda):
    if cos_phi > 1:
        cos_phi = 1
    elif cos_phi<-1:
        cos_phi = -1
    return np.exp(-Lambda*math.acos(cos_phi))


def fun_anisotropy_linear(cos_phi, Lambda):
    # print(type((np.pi-Lambda*math.acos(cos_phi))/np.pi))
    return max((np.pi-Lambda*math.acos(cos_phi))/np.pi, 0)


def fun_sparseness_legacy(db_ij, cos_phi, spa_lambda):
    ani = fun_anisotropy_linear(cos_phi, spa_lambda)
    if ani == 0:
        return 1000
    else:
        return db_ij / ani


def fun_sparseness_modified(db_ij, cos_phi, spa_lambda):
    return db_ij * (2 - fun_anisotropy_linear(cos_phi, spa_lambda))


def correct_angle_range(theta):
# keep angle range between -pi and pi
    if theta < -np.pi:
        theta = theta + 2*np.pi
    elif theta > np.pi:
        theta = theta - 2*np.pi
    return theta


def closest_point_on_line_segments(x, line_segments):
    """
    find the closest point from P to any of the line segments
    see https://stackoverflow.com/questions/10983872/distance-from-a-point-to-a-polygon

    :param P:
    :param line_segments:
    :return:
    """
    points = []
    dists = []
    for p1, p2 in line_segments:
        vec21 = p2 - p1
        r  = np.dot(vec21.transpose(), x - p1)
        # print('r=', r)
        r = r / (np.linalg.norm(p2 - p1) ** 2)

        # print('r=', r)
        if r < 0:
            dists.append(np.linalg.norm(x - p1))
            points.append(p1)
        elif r > 1:
            dists.append(np.linalg.norm(p2 - x))
            points.append(p2)
        else:
            dists.append(np.sqrt(np.linalg.norm(x - p1) ** 2 - (r * np.linalg.norm(p2 - p1)) ** 2))
            points.append(p1 + r * (p2 - p1))

    return points[dists.index(min(dists))]


def fun_lim_v(sparseness, F_veh_magnitude, beta_v_S, S_v_0, beta_v_F, F_v_0, v_max, v_normal, v_dense):
    v_lim = min(beta_v_S * max(sparseness - S_v_0, 0), v_normal - v_dense) + v_dense
    v_add = min(beta_v_F * max(F_veh_magnitude - F_v_0, 0), v_max - v_normal)
    return v_lim + v_add, v_lim, v_add


def fun_lim_a(sparseness, F_veh_magnitude, beta_a_S, S_a_0, beta_a_F, F_a_0, a_max, a_normal, a_dense):
    # force calculation
    a_lim = min(beta_a_S * max(sparseness - S_a_0, 0), a_normal - a_dense) + a_dense
    a_add = min(beta_a_F * max(F_veh_magnitude - F_a_0, 0), a_max - a_normal)
    return a_lim + a_add, a_lim, a_add


def fun_des_vd(ego, des, des_vd, des_sigma):
    des = np.array(des).reshape(2, 1)
    ego = np.array(ego).reshape(2, 1)
    r_ped2des = des - ego
    d_ped2des = np.linalg.norm(r_ped2des)
    vec_vd = des_vd * (des - ego) / np.sqrt(d_ped2des ** 2 + des_sigma)
    return vec_vd


def fun_des_beta(f_vehs, F_1, F_2):
    return max(min(1 / (F_2 - F_1) * (F_2 - np.linalg.norm(f_vehs)), 1), 0)
# closest_point_on_line_segments(np.array([[3], [4]]), [(a, b), (b*3, a+b)])
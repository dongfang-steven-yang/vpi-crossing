
class TrackerSingle:

    def __init__(self, dt, t0):
        self.t = t0
        self.dt = dt
        self.traj_t = []
        self.traj_class = []
        self.traj_pos = []
        self.active = False

    def update(self, obj_class, obj_pos):

        if obj_class is not None:
            self.active = True
            self.traj_t.append(self.t)
            self.traj_class.append(obj_class)
            self.traj_pos.append(obj_pos)
        else:
            self.active = False
            self.traj_t = []
            self.traj_class = []
            self.traj_pos = []

        self.t = self.t + self.dt
        return self.active

    def get_tracking(self, discard_tracking=False, history_length=1):
        """

        :param discard_tracking: indicate if imitate Uber accident
        :param history_length: in seconds
        :return:
        """

        len_history = len(self.traj_class)
        if discard_tracking:
            # forgot history tracking if classification changes --------------------------------------
            traj_pos_backward = []
            traj_class_backward = []
            i = 1 # index starting from end, loop is running backward
            c = True # if it is continually detected
            while c and i-1<len_history and i-1<history_length:
                traj_pos_backward.append(self.traj_pos[-i])
                traj_class_backward.append(self.traj_class[-i])
                # update condition
                if i == len_history or i == history_length:  # last data visited
                    c = False
                else:
                    if self.traj_class[-i] != self.traj_class[-(i+1)]:
                        c = False
                    i = i + 1
        else:
            # keep certain length of tracking history --------------------------------------------------
            len_history_requested = int(history_length / self.dt)
            if len_history < len_history_requested:
                len_history_requested = len_history
            traj_pos_backward = self.traj_pos[-len_history_requested:]
            traj_class_backward = self.traj_class[-len_history_requested:]
            traj_pos_backward.reverse()
            traj_class_backward.reverse()


        return traj_class_backward, traj_pos_backward, self.t - self.dt

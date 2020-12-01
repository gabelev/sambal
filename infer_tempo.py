import cv2
import math

import numpy as np



class InferTempo():

    def __init__(
            self,
            video_file_path,
    ):
        self.video_location = video_file_path


    def faircloth_tempo(self, flow: list) -> float:
        """

        an implementation of ryan faircloth's visual tempo detection algorithm.
        https://stars.library.ucf.edu/cgi/viewcontent.cgi?article=4582&context=etd

        args:
            flow (list): optical flow output from cv2.calcopticalflowfarneback

        returns:
            movement (float): angle of movement vectors of the optical flow per each frame
        """

        # slicing an numpy array is much nicer
        np_flow = np.array(flow)
        theta = np.arctan(np.mean(np_flow[..., 0]) / np.mean(np_flow[..., 1]))

        if (np.pi / 2 <= theta < np.pi):
            return (np.pi - theta)
        if (np.pi <= theta < 1.5 * np.pi):
            return (3 * np.pi - theta)
        return theta


    def calculate_motion_vector_angles(
            self,
            optical_flow_params: tuple = (0.5, 3, 15, 3, 5, 1.2, 0),
            convolve_param: int = 10
    ) -> np.array:
            """
            Run the optical flow Farneback algorithm pointed at a video file

            Args:
                video_location (string): location of the video file
                flow_params(list): "default" params for openCV Farneback

            Returns:
                movement (list): angles for movement vectors of the optical flow per each frame

            Raises:
                FileNotFoundError
            """
            capture = cv2.VideoCapture(self.video_location)
            success, previous_frame = capture.read()
            if not success:
                raise FileNotFoundError("Video file not found")

            previous_frame_greyscale = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

            # A useful given
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

            # initialize optical flow and angles lists
            motion_angles = list()

            for _ in range(frame_count - 1):
                _, current_frame = capture.read()
                current_frame_grayscale = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(
                    previous_frame_greyscale,
                    current_frame_grayscale,
                    None,
                    *optical_flow_params
                )
                motion_angles.append(self.faircloth_tempo(flow))
                previous_frame_greyscale = current_frame_grayscale

            return np.convolve(motion_angles, np.ones(convolve_param), mode='full')

    def build_beat_candidates(self, motion_angles: list, threshold: float = 5.0) -> list:
        """
        Filter only the strongest beat candidates and their location

        Args:
            motion_angles (list): output of the calculate_faircloth_tempo, a list of angles of motion vectors between frames
            threshold (int): barn door filter threshold

        Returns:
            beat_candidates (list): a filtered list of likely candidates

        Raises:
            None
        """
        return [(i, j) for i, j in enumerate(motion_angles) if j > threshold]


    def agent_factory(
            self,
            interval: int = 0,
            prediction: float = 0.0,
            history: list = [],
            score: float = 0.0,
            penalties: int = 0,
            matches: int = 0
    ) -> dict:
        """
        A factory for agents in the tempo inference algo with defaults.

        """
        return dict(
            interval=interval,
            prediction=prediction,
            history=history,
            score=score,
            penalties=penalties,
            matches=matches,
        )


    def infer_faircloth_tempo(
            self,
            candidates: list,
            num_frames: int,
            tempos: list = tuple(range(90,180)),
    ) -> float:
        """
        Implementation of the Faircloth visual tempo algorithim
        https://stars.library.ucf.edu/cgi/viewcontent.cgi?article=4582&context=etd

        args:
            candidates (list of tuples): list of tuples of (frame_num, motion_vector_angle)
            num_frames (int): number of frames in video
            tempos (list): hypothesis tempos

        returns:
            tempo (float): tempo infered from the beat candidates
            score (float): probabilistic score for visual tempo
            beats (list): array of frame numbers corresponding to the tracked visual beats

        raises:
            None
        """

        # initialize base params
        startup_period = math.ceil(0.75*num_frames)
        outer_tolerance_pre = 0.25
        outer_tolerance_post = 0.25
        inner_tolerance = 8
        total_salience = sum([i[1] for i in candidates])
        num_candidates = len(candidates)
        # use the agent_factory method to build out this list
        agents = []

        for tempo in tempos:
            for vote in candidates:
                if vote[0] < startup_period:
                    agent = agent_factory(
                        tempo,
                        vote[0] + tempo,
                        [vote],
                        vote[1]/total_salience,
                        0,
                        1,
                    )
                    agents.append(agent)

        for vote in candidates:
            new_agents = []
            for agent in agents:
                pre_tolerance = math.ceil(outer_tolerance_pre*agent['interval'])
                post_tolerance = math.ceil(outer_tolerance_post*agent['interval'])
                timeout = agent['interval'] + post_tolerance
                while (vote[0] - agent['history'][len(agent['history'])-1][0]) > timeout:
                    new_beat_onset = agent['history'][len(agent['history'])-1][0] + agent['interval']
                    agent['history'].append((new_beat_onset, 0))
                    agent['penalties'] += 1
                    agent['prediction'] += agent['interval']

                tolerance_width = pre_tolerance + post_tolerance

                while ((agent['prediction'] + post_tolerance) < vote[0]):
                    agent['prediction'] += agent['interval']

                if (agent['prediction'] - pre_tolerance <= vote[0]) \
                and (vote[0] <= agent['prediction'] + post_tolerance):
                    if abs(agent['prediction'] - vote[0]) > inner_tolerance:
                        new_agents.append(agent)

                    error = vote[0] - agent['prediction']
                    real_error = error/tolerance_width
                    agent['matches'] += 1
                    agent['interval'] += real_error
                    agent['prediction'] = vote[0] + agent['interval']
                    agent['history'].append(vote)
                    score_inc = (vote[1]/total_salience)*(1-abs(real_error))
                    agent['score'] += score_inc
            agents.extend(new_agents)

        for agent in agents:
            agent['score'] = agent['score']*((1.0 - ((agent.get('penalties')*agent.get('interval'))/num_frames))\
            *agent.get('matches')/num_candidates)

        max_score = 0
        index_max = 100000000000
        for index, agent in enumerate(agents):
            if agent.get('score') > max_score:
                max_score = agent.get('score')
                index_max = index
        return agents[index_max]
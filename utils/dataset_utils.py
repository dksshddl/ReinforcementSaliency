import numpy as np
import cv2

def embed_frame(observation, n_samples=8, width=224, height=224, n_channels=3):
    if len(observation) > n_samples:
        observation = observation[:n_samples]
    elif len(observation) < n_samples:
        embed = np.zeros(shape=(width, height, n_channels))
        for _ in range(n_samples - len(observation)):
            observation = np.concatenate([observation, [embed]])
    return observation


def read_whole_video(cap, fx=0.3, fy=0.3):
    video = []
    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, dsize=(0, 0), fx=fx, fy=fy)
            # cv2.imshow("test",frame)
            video.append(frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        else:
            cap.release()
            break
    # cv2.destroyAllWindows()
    return video

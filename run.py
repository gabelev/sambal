#!/usr/bin/env python

import sys

from infer_tempo import InferTempo

if __name__ == "__main__":

    try:
        video = sys.argv[1]
    except IndexError:
        # use the default in repo
        video = "media/office_dance_vid.mp4"

    infer_tempo = InferTempo(video)
    tempo, confidence = infer_tempo.run()
    print(f"Tempo is: {tempo}, with a confidence of {confidence * 100}")

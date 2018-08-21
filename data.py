import numpy

def normalizer(video):

	centered = video - video.mean()

	tightened = centered / centered.var()

	return tightened



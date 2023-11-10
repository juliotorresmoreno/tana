
from pipe.Pipeline import Response, Arguments
from pipe.reinformer.base import Reinformer
from sentence_transformers import CrossEncoder
from transformers import pipeline
from decouple import config

CHECKPOINT = config('HALLUCINATION_EVALUATION_MODEL')


class Hallucination(Reinformer):
    task = "reinformer"
    encoder = pipeline('text-classification', CHECKPOINT)

    def __init__(self):
        super().__init__(self.task)

    def invoke(self, args: Arguments) -> Response:
        if args.release != None and args.release != '':
            result = self.encoder.predict(args.question + ' [SEP] ' + args.release)
            if (result[0]['score'] > .5):
                return Response(result='I so sorry. I cannot answer this.')

        return Response(result=args.release, execution_time=0)

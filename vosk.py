from vosk import Model, KaldiRecognizer
import pyaudio
from nltk.metrics.distance import edit_distance
from std_msgs.msg import UInt8
import rospy
import numpy as np

class Vosk_pub():

    def __init__(self) -> None:
        self.pub = rospy.Publisher('alice', UInt8, queue_size=10)

        self.model = Model(r"/home/wladimir/catkin_ws/src/ISR_ros_vosk/vosk_ru_small/vosk-model-small-ru-0.22")
        self.rec = KaldiRecognizer(self.model, 16000)
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1, 
            rate=16000,
            input=True, 
            frames_per_buffer=1000
        )

    def inference(self):
        self.stream.start_stream()

        while not rospy.is_shutdown():
            data = self.stream.read(1000)
            if self.rec.AcceptWaveform(data):
                coms = self.rec.Result()[14:-3].split()
                if coms is not None:
                    self.pub.publish(coms)

if __name__ == '__main__':
    rospy.init_node('vosk_pub')
    node = Vosk_pub()
    node.inference()

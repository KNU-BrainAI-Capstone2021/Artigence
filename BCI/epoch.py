import sys
import numpy
import numpy as np

sys.path.append("C:/Users/PC/Desktop/BCI")
print(sys.path)
import multi_classification
import Write

class MyOVBox(OVBox):
	def __init__(self):
		OVBox.__init__(self)
		self.signalHeader = None
		self.signalBuffer = list()
		self.network_input = list()
	def process(self):
		for chunkIdx in range(len(self.input[0])):
			print(chunkIdx)
			if (type(self.input[0][chunkIdx]) == OVSignalHeader):
				self.signalHeader = self.input[0].pop()
				outputHeader = OVSignalHeader(self.signalHeader.startTime, self.signalHeader.endTime,
											  self.signalHeader.dimensionSizes,
											  self.signalHeader.dimensionLabels,
											  self.signalHeader.samplingRate)

				self.output[0].append(outputHeader)
			elif (type(self.input[0][chunkIdx]) == OVSignalBuffer):
				chunk = self.input[0].pop()
				numpyBuffer = numpy.array(chunk)
				# numpyBuffer = numpyBuffer.mean(axis=0)
				print(chunk.startTime, chunk.endTime)
				chunk = OVSignalBuffer(chunk.startTime, chunk.endTime, numpyBuffer.tolist())
				
				self.output[0].append(chunk)
				if chunk.endTime % 10 == 0.:
					self.signalBuffer.append(chunk)
					for i in range(len(self.signalBuffer)):
						signal = self.signalBuffer.pop(0)

						signal = numpy.array(signal)
						signal = np.reshape(signal, (28, -1))

						if i ==0:
							self.network_input = signal.tolist()
						else:
							self.network_input = np.concatenate((np.array(self.network_input),signal),axis = 1).tolist()
						#print(len(self.signalBuffer.pop(0)))
						#self.output[0].append(self.signalBuffer.pop(0))
					multi_classification.Network(self.network_input)
					self.network_input.clear()
				else:
					print("in")
					print("input chunck len",len(chunk))
					print(len(self.signalBuffer))
					self.signalBuffer.append(chunk)
			elif (type(self.input[0][chunkIdx]) == OVSignalEnd):
				self.output[0].append(self.input[0].pop())
				

	def uninitialize(self):
		print('Python uninitialize function started')
		return
box = MyOVBox()

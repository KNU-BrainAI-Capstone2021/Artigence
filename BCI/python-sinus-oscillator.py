import numpy

class MyOVBox(OVBox):
	def __init__(self):
		OVBox.__init__(self)
		self.nChannel = 0
		self.sampling = 0
		self.epochSampleCount = 0
		self.startTime = 0.
		self.endTime = 0.
		self.dimensionSizes = list()
		self.dimensionLabels = list()
		self.timeBuffer = list()
		self.signalBuffer = None
		self.signalHeader = None
		
	def initialize(self):
		self.nChannel = int(self.setting['Channel count'])
		self.sampling = int(self.setting['Sampling frequency'])
		self.epochSampleCount = int(self.setting['Generated epoch sample count'])
		
		#creation of the signal header
		for i in range(self.nChannel):
			self.dimensionLabels.append( 'Sinus'+str(i) )
		self.dimensionLabels += self.epochSampleCount*['']
		self.dimensionSizes = [self.nChannel, self.epochSampleCount]
		self.signalHeader = OVSignalHeader(0., 0., self.dimensionSizes, self.dimensionLabels, self.sampling)
		self.output[0].append(self.signalHeader)
		
		#creation of the first signal chunk
		self.endTime = 1.*self.epochSampleCount/self.sampling
		self.signalBuffer = numpy.zeros((self.nChannel, self.epochSampleCount))
		self.updateTimeBuffer()
		self.updateSignalBuffer()
		
	def updateStartTime(self):
		self.startTime += 1.*self.epochSampleCount/self.sampling
		
	def updateEndTime(self):
		self.endTime = float(self.startTime + 1.*self.epochSampleCount/self.sampling)
	
	def updateTimeBuffer(self):
		self.timeBuffer = numpy.arange(self.startTime, self.endTime, 1./self.sampling)
		
	def updateSignalBuffer(self):
		for rowIndex, row in enumerate(self.signalBuffer):
			self.signalBuffer[rowIndex,:] = 100.*numpy.sin( 2.*numpy.pi*(rowIndex+1.)*self.timeBuffer )
			
	def sendSignalBufferToOpenvibe(self):
		start = self.timeBuffer[0]
		end = self.timeBuffer[-1] + 1./self.sampling
		bufferElements = self.signalBuffer.reshape(self.nChannel*self.epochSampleCount).tolist()
		self.output[0].append( OVSignalBuffer(start, end, bufferElements) )
	
	def process(self):
		start = self.timeBuffer[0]
		end = self.timeBuffer[-1]
		if self.getCurrentTime() >= end:
			self.sendSignalBufferToOpenvibe()
			self.updateStartTime()
			self.updateEndTime()
			self.updateTimeBuffer()
			self.updateSignalBuffer()

	def uninitialize(self):
		end = self.timeBuffer[-1]
		self.output[0].append(OVSignalEnd(end, end))				

box = MyOVBox()

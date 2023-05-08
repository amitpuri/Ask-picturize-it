import logging
from pymongo import MongoClient


class MongoFormatter(logging.Formatter):
    def format(self, record):
        data = record.__dict__.copy()

        if record.args:
            msg = record.msg % record.args
        else:
            msg = record.msg

        data.update(
            message=msg,
            args=tuple(unicode(arg) for arg in record.args)
        )
        if 'exc_info' in data and data['exc_info']:
            data['exc_info'] = self.formatException(data['exc_info'])
        return data


class MongoHandler(logging.Handler):
	@classmethod
	def to(cls, db, collection, connection_string):
        	return cls(collection, db, connection_string)


	def __init__(self, db, collection, connection_string):
		logging.Handler.__init__(self, logging.ERROR)
		self.client = MongoClient(connection_string)
		self.collection =  self.client[db][collection]
		self.formatter = MongoFormatter()

	def emit(self, record):
		try:
			self.collection.insert_one(self.format(record))
		except InvalidDocument as e:
			logging.error("Loggging Error: %s", e.message,exc_info=True)
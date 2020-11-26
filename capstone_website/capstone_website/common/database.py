
class Database(object):

    @staticmethod
    def initialize():       # Initializes Database (Mongodb must be already running on system)
        client = pymongo.MongoClient(Database.URI)
        Database.DATABASE = client['fullstack']

    @staticmethod
    def insert(collection, data):       # Inserts new record in db.collection (data must be in JSON)
        Database.DATABASE[collection].insert(data)

    @staticmethod
    def find(collection, query):        # Returns all records from db.collection matching query
        return Database.DATABASE[collection].find(query)        # query must be in JSON

    @staticmethod
    def find_one(collection, query):    # Returns fist record from db.collection matching query
        return Database.DATABASE[collection].find_one(query)    # query must be in JSON

    @staticmethod
    def update(collection, query, data):    # Modifies record matching query in db.collection
        # (upsert = true): creates a new record when no record matches the query criteria
        Database.DATABASE[collection].update(query,data,upsert = True)

    @staticmethod
    def remove(collection, query):      # Deletes record from db.collecion
        Database.DATABASE[collection].remove(query)

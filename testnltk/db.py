import mysql.connector

class db_connection:
    def __init__(self,user='root',password='mining',host='127.0.0.1',database='masterthesis',usepure=True):
        self.config={
        'user': user,
        'password': password,
        'host': host,
        'database': database,
        'raise_on_warnings': True,
        'use_pure': usepure,
    }
    def connect(self):
        self.connection = mysql.connector.connect(**self.config)
    def disconnect(self):
        self.connection.close()
class db_operator:
    def __init__(self,db_con):
        self.dbconnection=db_con
    def executeSelection(self,query):
        cursor = self.dbconnection.connection.cursor()
        cursor.execute(query)
        rows=[]
        for t in cursor:
            rows.append(t)
        cursor.close()
        return rows

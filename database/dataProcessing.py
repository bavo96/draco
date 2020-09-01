import os, sys
from pathlib import Path

# SRC_DIR = str(Path(os.getcwd()).parent.parent)
SRC = str(Path(os.getcwd()).parent)
sys.path.insert(0, SRC)
#print(sys.path)

import mysql.connector
from mysql.connector import errorcode
from mysql.connector import Error
from tzlocal import get_localzone
import conf.conf as cfg
import json
from datetime import datetime
import re
import logging

create_time = datetime.now(get_localzone()).strftime('%y-%m-%d %H:%M:%S')

logger = logging.getLogger(__name__)


class MySQL():
    def get_mysql_connection(self, db_info):
        try:
            dbconfig = {'user': db_info['DB_MYSQL_USERNAME'], 'password': db_info['DB_MYSQL_PASSWORD'], \
            'host': db_info['DB_MYSQL_HOST'], 'port': db_info['DB_MYSQL_PORT'], 'database': db_info['DB_MYSQL_NAME']}
            conn = mysql.connector.connect(pool_name = dbconfig['database'], pool_size = 4, **dbconfig)
            return conn
        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                logger.error("User name or password are denied.")
            elif err.errno == errorcode.ER_BAD_DB_ERROR:
                logger.error("Database does not exist.")
            else:
                logger.error("%s" %(err))
        return None

    def execute_mysql(self, conn, query, value=None):
        print(query)
        cursor = conn.cursor(dictionary=True)
        if value is not None:
            cursor.execute(query, value)
        else:
            cursor.execute(query)
        return cursor

class dataStructure:
    def __init__(self):
        self.DB = None

    def get_connection(self, db, db_info):
        if db == 'MySQL':
            self.DB = MySQL()
            return self.DB.get_mysql_connection(db_info)
        else:
            raise ValueError('DB is unsupported.')
    
    def get_data_mysql(self, query, db_info):
        conn = self.get_connection('MySQL', db_info)
        try:
            if conn:
                cursor = self.DB.execute_mysql(conn, query)
                while True:
                    data = cursor.fetchmany(cfg.NUM_ROW)
                    if data:
                        yield data
                    else:
                        break
            

        except mysql.connector.Error as error:
            #logger.error('Failed to get data: {}'.format(error))
            raise error
        except Exception as E:
            raise E
        finally:
            if conn.is_connected():
                cursor.close()
                conn.close()
            else:
                logger.error("Can't connect to DB.")

    def check_data_exist(self, db_info, db_table, where_cond, similar=False):
        conn = self.get_connection('MySQL', db_info)
        data_existed = False
        try:
            if conn:
                if similar:
                    where_condition = " and ".join([(k + " like " + "'%" + str(v) + "%'") for k, v in where_cond.items()])
                else:
                    where_condition = " and ".join([(k + "=" + "'" + str(v) + "'") for k, v in where_cond.items()])
                query_exist = "SELECT * FROM {} WHERE {}".format(db_table, where_condition)
                cursor_exist = self.DB.execute_mysql(conn, query_exist)
                record = cursor_exist.fetchall()
                if record:
                    data_existed = record
                cursor_exist.close()
        except mysql.connector.Error as error:
            data_existed = None
            logger.error('MySQL error: {}'.format(error))
        except Exception as E:
            data_existed = None
            logger.error('Error: {}'.format(E))
        finally:
            if conn.is_connected():
                conn.close()
            else:
                data_existed = None
                logger.info("Can't connect to DB.")
            return data_existed

    def write_data_mysql(self, data, db_info, db_table, updated_data=None):
        '''Log prediction.
        Args:
            data: data needs to insert to database, dictionary type.
            db_info: information of database, dictionary type.
            db_table: table in MySQL, string type.
        '''
        conn = self.get_connection('MySQL', db_info)
        #print(conn)
        try:
            if conn:
                field, value = zip(*[(k, str(v)) for k, v in data.items()])
                field, value = ",".join(field), ",".join(value)
                if updated_data:
                    updated_data = ",".join([(k + "=" + str(v)) for k, v in updated_data.items()])
                    query = "INSERT INTO {}({}) VALUES ({}) ON DUPLICATE KEY UPDATE {}".format(db_table, field, value, updated_data)
                else:
                    query = "INSERT INTO {}({}) VALUES ({})".format(db_table, field, value) 
                cursor = self.DB.execute_mysql(conn, query)
                cursor.close()
                conn.commit()
                logger.info("Data's added to MySQL.")

        except mysql.connector.Error as error:
            #logger.error('Failed to insert data to DB: {}'.format(error))
            raise error
        except Exception as E:
            #logger.error('Error: {}'.format(E))
            raise E
        finally:
            if conn.is_connected():
                conn.close()
            else:
                logger.error("Can't connect to DB.")

    def update_data_mysql(self, updated_data, db_info, db_table, where_condition):
        conn = self.get_connection('MySQL', db_info)
        try:
            if conn:
                set_data = ",".join([(k + "=" + str(v)) for k, v in updated_data.items()])
                where_condition = " and ".join([(k + "=" + str(v)) for k, v in where_condition.items()])
                query = "UPDATE {} SET {} WHERE {}".format(db_table, set_data, where_condition)
                cursor = self.DB.execute_mysql(conn, query)
                cursor.close()
                conn.commit()
                logger.info("Data's updated to MySQL.")
                return True
        except mysql.connector.Error as error:
            #logger.error('Failed to insert data to DB: {}'.format(error))
            raise error
        except Exception as E:
            #logger.error('Error: {}'.format(E))
            raise E
        finally:
            if conn.is_connected():
                conn.close()
            else:
                logger.error("Can't connect to DB.")

if __name__ == "__main__":
    ds = dataStructure()
    res = {}
    res['image_id'] = 123
    res['face_vector'] = "\"[]\""
    res['bib_code'] = "\"B123\""
    res['validation_bib_code'] = "\"\""
    ds.write_data_mysql(res, cfg.MYSQL, cfg.DB_MYSQL_PREDICTION_TABLE)

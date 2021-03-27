import pymysql
from sklearn.tree import DecisionTreeRegressor


def dataBase():
        connection=pymysql.connect(
            host='localhost',
            user='root',
            password='Domy0428!',
            db='dataset'
        )

        # Connessione al database
        cursor=connection.cursor()
        print('Connessione stabilita!')
        #Eseguo query 1
        cursor.execute('select danceability,liveness from training;')
        #Estraggo valori dalla query 1
        Training=cursor.fetchall()
        # Eseguo query 2
        cursor.execute('select emotion from training;')
        # Estraggo valori dalla query 2
        TrainingE=cursor.fetchall()
        #Costruisco l'albero di regressione
        regressore=DecisionTreeRegressor()
        #Addestro l'albero di regressione
        regressore.fit(Training,TrainingE)
        #Eseguo la query 3
        cursor.execute('select danceability,liveness from songs')
        # Estraggo valori dalla query 3
        Test=cursor.fetchall()
        #Effettuo predizione
        predizione=regressore.predict(Test)
        cursor.execute('select count(liveness) from songs')
        i=cursor.fetchone()
        j=int(i[0])
        k=0
        #aggiorna il dataset con le emozioni predette per ogni canzone
        while(k<=j-1):
            sql='UPDATE songs SET emotion=%s WHERE Danceability=%s AND Liveness=%s;'
            val=(predizione[k], Test[k][0], Test[k][1])
            cursor.execute(sql, val)
            connection.commit()
            k=k+1











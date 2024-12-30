# Replicação + Sharding + Fail Over = MongoDB

----------------------------------
# Prática
----------------------------------

## 1 - Vamos criar uma rede para instalarmos o cluster

```bash
docker network create net-cluster-mongo

docker network ls
```

## 2 - Vamos criar os 3 config servers:


```bash
docker run --name srv-mongo-config01 --net net-cluster-mongo -d mongo mongod --configsvr --replSet configserver --port 27017

docker run --name srv-mongo-config02 --net net-cluster-mongo -d mongo mongod --configsvr --replSet configserver --port 27017

docker run --name srv-mongo-config03 --net net-cluster-mongo -d mongo mongod --configsvr --replSet configserver --port 27017
```

## 3 - Vamos configurar o primeiro nó como "nó mestre":

```bash
docker exec -it srv-mongo-config01 mongosh

rs.initiate (
	{
		_id: "configserver", 	configsvr:true, 	version: 1, 	members:
		[
			{ _id: 0, host: "srv-mongo-config01:27017" },
			{ _id: 1, host: "srv-mongo-config02:27017" },
			{ _id: 2, host: "srv-mongo-config03:27017" }
		]
	}
)

exit
```

## 4 - Vamos configurar 3 clusters de Shards Servers. Primeiro o Shard 1:

```bash
docker run --name srv-shard-node01 --net net-cluster-mongo -d mongo mongod -port 27018 --shardsvr --replSet shard01

docker run --name srv-shard-node02 --net net-cluster-mongo -d mongo mongod -port 27018 --shardsvr --replSet shard01

docker run --name srv-shard-node03 --net net-cluster-mongo -d mongo mongod -port 27018 --shardsvr --replSet shard01
```

## 5 - Agora o Shard 2:

```bash
docker run --name srv-shard-node04 --net net-cluster-mongo -d mongo mongod -port 27019 --shardsvr --replSet shard02

docker run --name srv-shard-node05 --net net-cluster-mongo -d mongo mongod -port 27019 --shardsvr --replSet shard02

docker run --name srv-shard-node06 --net net-cluster-mongo -d mongo mongod -port 27019 --shardsvr --replSet shard02
```

## 6 - E agora o Shard 3:

```bash
docker run --name srv-shard-node07 --net net-cluster-mongo -d mongo mongod -port 27020 --shardsvr --replSet shard03

docker run --name srv-shard-node08 --net net-cluster-mongo -d mongo mongod -port 27020 --shardsvr --replSet shard03

docker run --name srv-shard-node09 --net net-cluster-mongo -d mongo mongod -port 27020 --shardsvr --replSet shard03
```

## 7 - Vamos inicializar o Server Node 01 do Shard 1, como principal e os Nodes 02 e 03 como réplicas:

```bash
docker exec -it srv-shard-node01 mongosh --port 27018

rs.initiate(
	{
		_id: "shard01", 	version: 1, 	members:
		[
			{ _id: 0, host : "srv-shard-node01:27018" },
			{ _id: 1, host : "srv-shard-node02:27018" },
			{ _id: 2, host : "srv-shard-node03:27018" }
		]
	}
)

exit
```

## 8 - Vamos inicializar o Server Node 04 do Shard 2, como principal e os Nodes 05 e 06 como réplicas:

```bash
docker exec -it srv-shard-node04 mongosh --port 27019

rs.initiate(
	{
		_id: "shard02", 	version: 1, 	members:
		[
			{ _id: 0, host : "srv-shard-node04:27019" },
			{ _id: 1, host : "srv-shard-node05:27019" },
			{ _id: 2, host : "srv-shard-node06:27019" }
		]
	}
)

exit
```

## 9 - Vamos inicializar o Server Node 07 do Shard 3, como principal e os Nodes 08 e 09 como réplicas:

```bash
docker exec -it srv-shard-node07 mongosh --port 27020

rs.initiate(
	{
		_id: "shard03", 	version: 1, 	members:
		[
			{ _id: 0, host : "srv-shard-node07:27020" },
			{ _id: 1, host : "srv-shard-node08:27020" },
			{ _id: 2, host : "srv-shard-node09:27020" }
		]
	}
)

exit
```

## 10 - Vamos criar nosso Router, atrelado aos config servers.

```bash
docker run -p 27017:27017 --name mongo-router --net net-cluster-mongo -d mongo mongos --port 27017 --configdb configserver/"srv-mongo-config01:27017, srv-mongo-config02:27017, srv-mongo-config03:27017" --bind_ip_all
```

## 11 - Vamos configurar o router:

```bash
docker exec -it mongo-router mongosh

sh.status()
```

## 12 - Vamos adicionar os Nós do Shard 01:

```bash
sh.addShard("shard01/srv-shard-node01:27018")

sh.addShard("shard01/srv-shard-node02:27018")

sh.addShard("shard01/srv-shard-node03:27018")

sh.status()
```

## 13 - Vamos adicionar os Nós do Shard 02:

```bash
sh.addShard("shard02/srv-shard-node04:27019")

sh.addShard("shard02/srv-shard-node05:27019")

sh.addShard("shard02/srv-shard-node06:27019")

sh.status()
```

## 14 - Vamos adicionar os Nós do Shard 03:

```bash
sh.addShard("shard03/srv-shard-node07:27020")

sh.addShard("shard03/srv-shard-node08:27020")

sh.addShard("shard03/srv-shard-node09:27020")

sh.status()
```

Tudo configurado! Hora de usar-mos!

## 15 - Vamos criar um Database e uma Coleção Particionada:

```bash
use DBSampleShard

db

sh.enableSharding("DBSampleShard")
```

### 15.1 - Ao criarmos a coleção, atribuímos a ela, o tipo de sharding (vamos escolher 'hash'):

```bash
sh.shardCollection( "DBSampleShard.exampleCollection", { "_id" : "hashed" } ) 

--> Veja como ficou a distribuição:

db.exampleCollection.getShardDistribution()
```

## 16 - Vamos adicionar alguns documentos na coleção:

```bash
db.exampleCollection.insertOne({"data_criacao_documento":new Date(), "descricao_solicitacao":'descrição 01'});

db.exampleCollection.insertOne({"data_criacao_documento":new Date(), "descricao_solicitacao":'descrição 02'});

db.exampleCollection.insertOne({"data_criacao_documento":new Date(), "descricao_solicitacao":'descrição 03'});

db.exampleCollection.insertOne({"data_criacao_documento":new Date(), "descricao_solicitacao":'descrição 04'});

db.exampleCollection.insertOne({"data_criacao_documento":new Date(), "descricao_solicitacao":'descrição 05'});

db.exampleCollection.insertOne({"data_criacao_documento":new Date(), "descricao_solicitacao":'descrição 06'});

db.exampleCollection.insertOne({"data_criacao_documento":new Date(), "descricao_solicitacao":'descrição 07'});

db.exampleCollection.insertOne({"data_criacao_documento":new Date(), "descricao_solicitacao":'descrição 08'});

db.exampleCollection.insertOne({"data_criacao_documento":new Date(), "descricao_solicitacao":'descrição 09'});
```

### 16.1 --> Veja como ficou a distribuição:

```bash
db.exampleCollection.getShardDistribution()
```

## 17 - Vamos adicionar mais alguns documentos na coleção:

```bash
db.exampleCollection.insertOne({"data_criacao_documento":new Date(), "descricao_solicitacao":'descrição 10'});

db.exampleCollection.insertOne({"data_criacao_documento":new Date(), "descricao_solicitacao":'descrição 11'});

db.exampleCollection.insertOne({"data_criacao_documento":new Date(), "descricao_solicitacao":'descrição 12'});

db.exampleCollection.insertOne({"data_criacao_documento":new Date(), "descricao_solicitacao":'descrição 13'});

db.exampleCollection.insertOne({"data_criacao_documento":new Date(), "descricao_solicitacao":'descrição 14'});

db.exampleCollection.insertOne({"data_criacao_documento":new Date(), "descricao_solicitacao":'descrição 15'});

db.exampleCollection.insertOne({"data_criacao_documento":new Date(), "descricao_solicitacao":'descrição 16'});

db.exampleCollection.insertOne({"data_criacao_documento":new Date(), "descricao_solicitacao":'descrição 17'});

db.exampleCollection.insertOne({"data_criacao_documento":new Date(), "descricao_solicitacao":'descrição 18'});

db.exampleCollection.insertOne({"data_criacao_documento":new Date(), "descricao_solicitacao":'descrição 19'});

db.exampleCollection.insertOne({"data_criacao_documento":new Date(), "descricao_solicitacao":'descrição 20'});

db.exampleCollection.insertOne({"data_criacao_documento":new Date(), "descricao_solicitacao":'descrição 21'});

db.exampleCollection.insertOne({"data_criacao_documento":new Date(), "descricao_solicitacao":'descrição 22'});

db.exampleCollection.insertOne({"data_criacao_documento":new Date(), "descricao_solicitacao":'descrição 23'});

db.exampleCollection.insertOne({"data_criacao_documento":new Date(), "descricao_solicitacao":'descrição 24'});

db.exampleCollection.insertOne({"data_criacao_documento":new Date(), "descricao_solicitacao":'descrição 25'});

db.exampleCollection.insertOne({"data_criacao_documento":new Date(), "descricao_solicitacao":'descrição 26'});

db.exampleCollection.insertOne({"data_criacao_documento":new Date(), "descricao_solicitacao":'descrição 27'});

db.exampleCollection.insertOne({"data_criacao_documento":new Date(), "descricao_solicitacao":'descrição 28'});

db.exampleCollection.insertOne({"data_criacao_documento":new Date(), "descricao_solicitacao":'descrição 29'});

db.exampleCollection.insertOne({"data_criacao_documento":new Date(), "descricao_solicitacao":'descrição 30'});
```

### 17.1 --> Veja como ficou a distribuição:

```bash
db.exampleCollection.getShardDistribution()
```

### 17.2 --> Vamos ver os documentos de todos os Shards:

```bash
db.exampleCollection.find().pretty()

exit
```

## 18 - Vamos ver os documentos do Shard 01:

```bash
docker exec -it srv-shard-node01 mongosh --port 27018

use DBSampleShard

db.exampleCollection.find().pretty()

exit
```

## 19 - Vamos testar a Replica Set:

### 19.1 - Pare o Nó 02 do Shard 01 (srv-shard-node02)!

### 19.2 - Vamos ao Router e vamos tentar adicionar mais coleções:

```bash
docker exec -it mongo-router mongosh

use DBSampleShard

db.exampleCollection.getShardDistribution()

db.exampleCollection.insertOne({"data_criacao_documento":new Date(), "descricao_solicitacao":'descrição 31'});

db.exampleCollection.insertOne({"data_criacao_documento":new Date(), "descricao_solicitacao":'descrição 32'});

db.exampleCollection.insertOne({"data_criacao_documento":new Date(), "descricao_solicitacao":'descrição 33'});

exit
```

### 19.3 - Volte a habilitar o Shard 2!

### 19.4 - Consulte os dados do Shard 2...

```bash
docker exec -it srv-shard-node02 mongosh --port 27018

use DBSampleShard

db.exampleCollection.find().pretty() #(talvez o dado não esteja aqui)

exit
```

Vamos fazer outros testes.

## 20 - Pare o Nó 02 do Shard 02 (srv-shard-node05)!

## 20.1 - Vamos nos conectar ao nó mestre do Shard 02 e vamos adicionar alguns documentos:

```bash
docker exec -it srv-shard-node04 mongosh --port 27019

use DBSampleShard

db.exampleCollection.insertOne({"data_criacao_documento":new Date(), "descricao_solicitacao":'descrição 34'});

db.exampleCollection.insertOne({"data_criacao_documento":new Date(), "descricao_solicitacao":'descrição 35'});

db.exampleCollection.insertOne({"data_criacao_documento":new Date(), "descricao_solicitacao":'descrição 36'});

db.exampleCollection.find().pretty()

exit
```

## 21 - Consulte o Nó 03 Shard 02 (srv-shard-node06):

## 21.1 - Localize os documentos 34, 35 e 36:

```bash
docker exec -it srv-shard-node06 mongosh --port 27019

use DBSampleShard

db.exampleCollection.find().pretty()

exit
```

## 22 - Volte a habilitar o Nó 02 do Shard 02 (srv-shard-node05) :

## 22.1 - Consulte os dados e tente localizar os documentos 34, 35 e 36:

```bash
docker exec -it srv-shard-node05 mongosh --port 27019

use DBSampleShard

db.exampleCollection.find().pretty()

exit
```

## 23 - Volte ao Router e consulte os dados:

```bash
docker exec -it mongo-router mongosh

use DBSampleShard

db.exampleCollection.getShardDistribution()

db.exampleCollection.find().pretty()

exit
```


Parabéns! Terminamos!
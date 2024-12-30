# Replicação + Loadbalancer + Fail Over = Postgres

----------------------------------
# Prática Replicação
----------------------------------

## 1 - Abrir o WSL

Entrar com o usuário root:

```bash
sudo su 
(senha)
```


Atualizar tudo:

```bash
apt update -y && apt upgrade -y

apt autoremove -y
```

## 2 - Instalar o PostgreSQL (última versão - 17)


```bash
apt install -y postgresql
```

## 3 - Alterar a senha do usuário 'postgres':

```bash
sudo -u postgres psql

ALTER USER postgres WITH PASSWORD 'postgres';

\q
```

## 4 - Alterar o arquivo pg_hba.conf:


### 4.1 - Vamos até a pasta da instalação do Postgres:

```bash

cd /etc/postgresql/17/main

```

Edite o arquivo pg_hba.conf com o editor NANO:

```bash
nano pg_hba.conf
```

### 4.2 - (buscar com CTRL + W) :

```bash
# Database administrative login by Unix domain socket
```

### 4.3 - Mudar disso : 

```bash
local all postgres [peer ou md5]
```

para (sem as chaves [...]) :

```bash
local all postgres [trust]
```

### 4.4 - Salve e saia do NANO:

```bash
CTRL + O (enter)
CTRL + X
```

## 5 - Alterar o arquivo postgresql.conf:

Fique na pasta do postgres e abra o arquivo postgresql.conf

```bash
nano postgresql.conf
```

### 5.1 - (buscar com CTRL + W) : 

```bash
#listen_addresses
```

### 5.2 - Mudar disso : 
```bash
#listen_addresses = 'localhost'
```

para:

```bash
listen_addresses = '*'
```

### 5.3 - Salve e saia do NANO:

```bash
CTRL + O (enter)
CTRL + X
```

## 6 - Reinicialize o serviço do Postgres:

```bash
service postgresql stop
service postgresql start
```

## 7 - Teste o usuário postgres com a nova senha (postgres):

```bash
psql -U postgres -W
(digite a senha).

\q  (para sair)
```

### 7.1 - Teste com PGAdmin do Windows


## 8 - Limpe resíduos de outras instalações:

```bash
rm -d -f -r /tmp/primary_db
rm -d -f -r /tmp/replica_db
```

## 9 - Altere para o usuário postgres e exponha o path do PostgreSQL:

```bash
sudo su - postgres

export PATH=$PATH:/usr/lib/postgresql/17/bin/
```

## 10 - Crie uma nova instância do Postgres chamada "primary" dessa maneira:

```bash
initdb -D /tmp/primary_db
```

## 11 - Edite as configurações do arquivo postgresql.conf da nova instância:

```bash
nano /tmp/primary_db/postgresql.conf
```

### 11.1 - (buscar com CTRL + W) : 

```bash
#listen_addresses
```

### 11.2 - Mudar de: 

```bash
#listen_addresses = 'localhost'

para

listen_addresses = '*'
```

```bash
de
#port = 5432

para

port = 5433
```

### 11.3 - Salve e saia do NANO:

```bash
CTRL + O (enter)
CTRL + X
```

## 12 - Inicialize a instância "primary" que criamos:

```bash
pg_ctl -D /tmp/primary_db start
```

### 12.1 - Testar com o PGAdmin

## 13 - Conecte-se na instância pelo psql e crie um usuário:

```bash
psql --port=5433 postgres

create user repuser replication;

\q
```

## 14 - Edite o arquivo pg_hba.conf da instância "primary" que criamos:

```bash
nano /tmp/primary_db/pg_hba.conf
```

### 14.1 - (buscar com CTRL + W) : 

```bash
# IPv4 local connections:
```

selecione a linha:
```bash
host    all             all             127.0.0.1/32            trust
```

e clique com o botão direito do mouse para copiar e va até o final da linha e dê um enter
clique com o botão direito do mouse para colar.

```bash
host    all             [all](mudar para -->) repuser             127.0.0.1/32            trust
```

### 14.2 - Salve e saia do NANO:

```bash
CTRL + O (enter)
CTRL + X
```

### 14.3 - Reinicialize a instância "primary":

```bash
pg_ctl -D /tmp/primary_db restart
```

## 15 - Crie a Réplica Secundária com o comando 'pg_basebackup':

```bash
pg_basebackup -h localhost -U repuser --checkpoint=fast -D /tmp/replica_db/ -R --slot=slot1 -C --port=5433
```

### 15.1 - Edite o arquivo postgresql.conf da "réplica":

```bash
cd /tmp/replica_db

nano postgresql.conf
```

### 15.2 - (buscar com CTRL + W) : 
```bash
port
```

### 15.3 - Mudar de: 

```bash
port = 5433

para

port = 5434
```

### 15.4 - Salve e saia do NANO:

```bash
CTRL + O (enter)
CTRL + X
```

## 15 - Inicialize a instância "réplica":

```bash
pg_ctl -D /tmp/replica_db start
```

### 15.1 - Testar com o PGAdmin

(criar tabelas e testar a replica)


----------------------------------

# Prática Load Balancer


## 1 - Vamos instalar o PG Pool 2:

```bash
exit (sair do usuário postgres)

sudo sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'

wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -

sudo apt-get update

sudo apt-get -y install pgpool2 libpgpool2 postgresql-17-pgpool2
```

## 2 - Pare o PG Pool 2 para podermos configurá-lo:

```bash

pgpool stop

cd /etc/pgpool2/

nano pgpool.conf
```

## 3 - Vamos alterar o arquivo "pgpool.conf":

### 3.1 - (buscar com CTRL + W) : 

```bash
listen_addresses
```

### 3.2 - Mudar: 

```bash
de:
#listen_addresses = 'localhost'

para:
listen_addresses = '*'

---

(buscar com CTRL + W) : # - Backend Connection Settings

mudar de:
#backend_hostname0 = 'host1' para

para:
backend_hostname0 = 'localhost'

---

mudar de:
#backend_port0 = 5432

para:
backend_port0 = 5433

---

mudar de:
#backend_weight0 = 1

para:
backend_weight0 = 0

---

mudar de:
#backend_data_directory0 = "/data"

para:
backend_data_directory0 = "/tmp/primary_db/"

---

mudar de:
#backend_hostname1 = 'host2' para

para:
backend_hostname1 = 'localhost'

---

mudar de:
#backend_port1 = 5433

para:
backend_port1 = 5434

---

mudar de:
#backend_data_directory1 = "/data"

para:
backend_data_directory1 = "/tmp/replica_db/"

---

(buscar com CTRL + W) : #log_statement

mudar de:
#log_statement = off

para:
log_statement = on

---

mudar de:
#log_per_node_statement = off

para:
log_per_node_statement = on

---

(buscar com CTRL + W) : #pid_file_name

mudar de:
#pid_file_name = '/var/run/postgresql/pgpool.pid'

para:
pid_file_name = 'pgpool.pid'

---

(buscar com CTRL + W) : #sr_check_user

mudar de:
#sr_check_user = 'nobody'

para:
sr_check_user = 'repuser'

---

(buscar com CTRL + W) : #health_check_period

mudar de:
#health_check_period = 0

para:
health_check_period = 10

---

mudar de:
#health_check_user = 'nobody'

para:
health_check_user = 'repuser'

```

### 3.4 - Salve e saia do NANO:

```bash
CTRL + O (enter)
CTRL + X
```


## 4 - Vamos acionar o pool :


### 4.1 - Volte com o usuário 'postgres' e execute o comando para inicializar o pool (dará erro):

```bash
sudo su - postgres
pgpool -n &
(Erro ---> Anotar o PID) [1] XXXXX
CTRL+C
```

### 4.2 - Crie o arquivo com o PID:

```bash
exit (---> sair do usuário postgres)

cd /etc/pgpool2
touch pgpool.pid
nano pgpool.pid (digitar o numero do processo)
```

### 4.3 - Salve e saia do NANO:

```bash
CTRL + O (enter)
CTRL + X
```

### 4.4 - Altere a propriedade do arquivo:

```bash
chown postgres:postgres pgpool.pid
```

### 4.5 - Execute novamente o pool:

```bash
sudo su - postgres

cd /etc/pgpool2

pgpool -n &
```

### 4.6 - Testar no PGAdmin

### 4.7 - Encerre o pool ---> Vamos para Fail Tolerance


----------------------------------
Fail Tolerance
----------------------------------


## 1 - Vamos modificar um pouco o 'postgresql.conf':

```bash
cd /tmp/primary_db/
nano postgresql.conf
```

### 1.1 - (buscar com CTRL + W) : 

```bash
#synchronous_commit
```

### 1.2 - Mudar: 

```bash
de:
#synchronous_commit = on

para:
synchronous_commit = remote_apply

---

(buscar com CTRL + W) : #synchronous_standby_names

mudar de:
#synchronous_standby_names = ''

para:
synchronous_standby_names = '*'

---

(buscar com CTRL + W) :  #wal_log_hints

mudar de:
#wal_log_hints = off

para:
wal_log_hints = on
```

### 1.3 - Salve e saia do NANO:

```bash
CTRL + O (enter)
CTRL + X
```

## 2 - Vamos reinicializar o postgres:

```bash
export PATH=$PATH:/usr/lib/postgresql/17/bin/
pg_ctl -D /tmp/primary_db restart

(CTRL+C se necessário)
```

## 3 - Vamos criar um script para automatizar a troca de servidores:

```bash
mkdir /tmp/pgsql
touch /tmp/pgsql/failover.sh
nano  /tmp/pgsql/failover.sh
```

copiar esse código e usar o botão direito para colar:

```bash
#! /bin/sh
cd /etc/pgpool2
pg_ctl promote -D /tmp/replica_db/
```

### 3.1 - Salve e saia do NANO:

```bash
CTRL + O (enter)
CTRL + X
```

### 3.2 - Torne o script executável:

```bash
chmod 777 /tmp/pgsql/failover.sh

exit (sair do usuário postgres)
```

## 4 - Vamos alterar o arquivo de configuração do pool ('pgpool.conf'):

```bash
nano /etc/pgpool2/pgpool.conf

(buscar com CTRL + W) : #failover_command

mudar de: 
#failover_command = ''

para:
failover_command = '/tmp/pgsql/failover.sh'
```

### 4.1 - Salve e saia do NANO:

```bash
CTRL + O (enter)
CTRL + X
```

## 5 - Vamos reinicializar o postgres:

```bash
sudo su - postgres

export PATH=$PATH:/usr/lib/postgresql/17/bin/

pgpool -n &
```

## 6 - Vamos testar, vamos parar o servidor primário:

```bash
pg_ctl stop -D /tmp/primary_db/
```

### 6.1 - Verificar se o dado continua sendo salvo no nó "réplica".

--------------------------
# Desinstalação
--------------------------

```bash
exit (sair do postgres)

sudo apt-get --purge -y remove pgpool2 libpgpool2 postgresql-17-pgpool2

sudo apt-get --purge -y remove postgresql postgresql-*

cd /etc/pgpool2

rm pgpool.pid

cd ..

rm -d pgpool2

rm -d -f -r /tmp/primary_db
rm -d -f -r /tmp/replica_db
```

Parabéns! Terminamos!
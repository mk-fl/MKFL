import docker, os

client = docker.from_env()
serv = client.containers.get('mkfl-flsrv-1')
ip = serv.attrs['NetworkSettings']['Networks']['mkfl_fl']['IPAddress']

with open('FL/certificates/certificate_docker.conf', "a+") as flclw:
     flclw.write("IP.3 = " + ip + '\n')

os.system("FL/certificates/generate_srv.sh")

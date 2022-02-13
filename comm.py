import base64
from time import sleep

import torch
from twisted.internet import reactor
from twisted.internet.protocol import Protocol, Factory, ReconnectingClientFactory
from twisted.protocols.basic import LineReceiver
from json import JSONDecodeError
import json
from io import BytesIO
import paramiko
from scp import SCPClient


class AlphaZeroDataBrokerProtocol(LineReceiver):
    CMD_PUT_PARAMETERS = 'PUT_PARAMS'
    CMD_GET_PARAMETERS = 'GET_PARAMS'
    CMD_ADD_SELF_PLAY_DATA = 'ADD_DATA'
    CMD_GET_SELF_PLAY_DATA = 'GET_DATA'

    STATUS_ERROR = 'ERR'
    STATUS_OK = 'OK'

    MAX_LENGTH = 16384 * 16

    def __init__(self, factory):
        self.factory = factory

    def lineReceived(self, msg):
        msg = msg.decode('ascii')
        try:
            msg = json.loads(msg)
            for expected in ['cmd', 'id']:
                if expected not in msg.keys():
                    raise ValueError('Expected key ' + expected + ' in msg, but got: ' + str(msg))
            if msg['cmd'] in [self.CMD_PUT_PARAMETERS, self.CMD_ADD_SELF_PLAY_DATA] and 'data' not in msg.keys():
                raise ValueError('Expected key data in msg, but got: ' + str(msg))
        except JSONDecodeError as e:
            print("Error while decoding: {}".format(e))
            self.sendLine(json.dumps({'status': self.STATUS_ERROR}).encode('ascii'))
            return
        resp = {'status': self.STATUS_OK, 'id': msg['id']}
        if msg['cmd'] == self.CMD_ADD_SELF_PLAY_DATA:
            print("Request to add self play data.")
            self.factory.self_play_data.append(msg['data'])
        elif msg['cmd'] == self.CMD_GET_SELF_PLAY_DATA:
            print("Request to get self play data. Sending.")
            items = min(8, len(self.factory.self_play_data))
            resp['data'] = self.factory.self_play_data[:items]
            self.factory.self_play_data = self.factory.self_play_data[items:]
        elif msg['cmd'] == self.CMD_PUT_PARAMETERS:
            print("Request to put parameters.")
            self.factory.parameters = msg['data']
        elif msg['cmd'] == self.CMD_GET_PARAMETERS:
            print("Request to send parameters")
            resp['data'] = self.factory.parameters
        resp = json.dumps(resp).encode('ascii')
        print(len(resp))
        self.sendLine(resp)

    def connectionMade(self):
        print("Got new connection.")

    def connectionLost(self, reason):
        print("Lost connection. Reason: {}".format(reason))


class AlphaZeroDataBrokerFactory(Factory):

    def __init__(self):
        self.self_play_data = []
        self.parameters = None

    def buildProtocol(self, addr):
        print('Connected to {}.'.format(addr))
        return AlphaZeroDataBrokerProtocol(self)


# === Client
class AlphaZeroClientProtocol(LineReceiver):

    MAX_LENGTH = 16384 * 256

    def __init__(self, factory):
        self.factory = factory
        self.factory.connection = self

    def send(self, cmd: str, request_id, data: str = None):
        msg = {'cmd': cmd, 'id': request_id}
        if data is not None:
            msg['data'] = data
        self.sendLine(json.dumps(msg).encode('ascii'))

    def lineReceived(self, msg):
        msg = msg.decode('ascii')
        try:
            msg = json.loads(msg)
        except JSONDecodeError as e:
            print("Client: Error while decoding: {}".format(e))
            print(msg)
            # self.factory.callback[msg.id]({'status': AlphaZeroDataBrokerProtocol.STATUS_ERROR})
            return
        if msg['status'] != 'OK':
            print("Error!")
        msg_id = int(msg['id'])
        callback = self.factory.callback[msg_id]
        callback(msg)


class AlphaZeroClientFactory(ReconnectingClientFactory):

    def __init__(self):
        self.connection = None
        self.next_id = 0
        self.callback = {}

    def startedConnecting(self, connector):
        print('Started to connect.')

    def get_id(self):
        res = self.next_id
        self.next_id += 1
        return res

    def buildProtocol(self, addr):
        print('Connected to {}.'.format(addr))
        print('Resetting reconnection delay')
        self.resetDelay()
        return AlphaZeroClientProtocol(self)

    def clientConnectionLost(self, connector, reason):
        print('Lost connection.  Reason:', reason)
        ReconnectingClientFactory.clientConnectionLost(self, connector, reason)

    def clientConnectionFailed(self, connector, reason):
        print('Connection failed. Reason:', reason)
        ReconnectingClientFactory.clientConnectionFailed(self, connector, reason)


class AlphaZeroClient:

    def __init__(self, host: str, port: int, user: str, passwd: str):
        super().__init__()
        self.factory = None
        self.host = host
        self.port = port
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(host, 22, user, passwd)
        self.ssh_client = client
        self.scp_client = SCPClient(self.ssh_client.get_transport())
        self.param_i = 0

    def _send(self, callback, cmd, data=None):
        while self.factory.connection is None:
            print("Waiting for connection.")
            sleep(0.5)
        req_id = self.factory.get_id()
        self.factory.callback[req_id] = callback
        self.factory.connection.send(cmd, req_id, data)

    @staticmethod
    def encode(obj):
        s = BytesIO()
        torch.save(obj, s)
        res = base64.b64encode(s.getvalue())
        res = res.decode('ascii')
        return res

    @staticmethod
    def decode(code):
        b = base64.b64decode(code.encode('ascii'))
        bio = BytesIO(b)
        bio.seek(0)
        return torch.load(bio, map_location='cpu')

    @staticmethod
    def decode_wrapper(callback):
        def wrapper(msg):
            if 'data' not in msg.keys():
                return callback(msg)
            code = msg['data']
            if isinstance(code, list):
                for i in range(len(msg['data'])):
                    msg['data'][i] = AlphaZeroClient.decode(msg['data'][i])
            else:
                msg['data'] = AlphaZeroClient.decode(msg['data'])
            return callback(msg)
        return wrapper

    def put_parameters(self, callback, parameters):
        remote_file = '/tmp/parameters_{}.pt'.format(self.param_i)
        local_file = '/tmp/parameters.pt'
        torch.save(parameters, local_file)
        self.scp_client.put(local_file, remote_file)
        self.param_i = (self.param_i + 1) % 3
        self._send(callback, AlphaZeroDataBrokerProtocol.CMD_PUT_PARAMETERS, remote_file)

    def read_parameters_wrapper(self, callback):
        def wrapper(msg):
            remote_file = msg['data']
            if msg['data'] is None:
                return callback(msg)
            local_file = '/tmp/parameters.pt'
            self.scp_client.get(remote_file, local_file)
            msg['data'] = torch.load('/tmp/parameters.pt', map_location='cpu')
            return callback(msg)
        return wrapper

    def get_parameters(self, callback):
        self._send(self.read_parameters_wrapper(callback), AlphaZeroDataBrokerProtocol.CMD_GET_PARAMETERS)

    def add_self_play_data(self, callback, self_play_data):
        self._send(callback, AlphaZeroDataBrokerProtocol.CMD_ADD_SELF_PLAY_DATA, self.encode(self_play_data))

    def get_self_play_data(self, callback):
        self._send(self.decode_wrapper(callback), AlphaZeroDataBrokerProtocol.CMD_GET_SELF_PLAY_DATA)

    def start(self):  # Blocks
        self.factory = AlphaZeroClientFactory()
        reactor.connectTCP(self.host, self.port, self.factory)
        reactor.run()

    def stop(self):
        self.ssh_client.close()
        reactor.stop()


def run_server(port):
    reactor.listenTCP(port, AlphaZeroDataBrokerFactory())
    reactor.run()

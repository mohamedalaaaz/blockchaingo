import json
import uuid
import random

# -----------------------------------------------------
# 1. Random ID generator (similar to your JS version)
# -----------------------------------------------------
def random_id(length=7, hex_mode=False):
    base = 16 if hex_mode else 36
    return ''.join(
        random.choice("0123456789abcdefghijklmnopqrstuvwxyz"[:base])
        for _ in range(length)
    )


# -----------------------------------------------------
# 2. Safe serializer (handles circular objects)
# -----------------------------------------------------
def safe_serialize(obj, seen=None, max_depth=10, depth=0):
    if seen is None:
        seen = set()

    if depth >= max_depth:
        return "MaxDepthExceeded"

    obj_id = id(obj)

    if obj_id in seen:
        return "[Circular]"

    seen.add(obj_id)

    # Primitive types
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj

    # Lists
    if isinstance(obj, list):
        return [
            safe_serialize(item, seen, max_depth, depth + 1)
            for item in obj
        ]

    # Dictionaries
    if isinstance(obj, dict):
        return {
            key: safe_serialize(value, seen, max_depth, depth + 1)
            for key, value in obj.items()
        }

    # Other Python objects
    result = {}
    for key in dir(obj):
        if key.startswith("__"):
            continue
        try:
            value = getattr(obj, key)
            result[key] = safe_serialize(value, seen, max_depth, depth + 1)
        except:
            pass

    return result


# -----------------------------------------------------
# 3. Async helper (similar to JS generator runner)
# -----------------------------------------------------
import asyncio

async def run_async_generator(gen):
    try:
        result = None
        while True:
            result = await gen.__anext__()
    except StopAsyncIteration:
        return result


# -----------------------------------------------------
# Example usage
# -----------------------------------------------------
if __name__ == "__main__":
    # Test random ID
    print("Random ID:", random_id())
    print("Hex ID:", random_id(7, True))

    # Test safe serializer
    a = {}
    b = {"ref": a}
    a["ref"] = b  # circular reference
    print("Serialized:", safe_serialize(a))

    # Async generator test
    async def sample_gen():
        yield 1
        yield 2
        yield 3

    async def test():
        value = await run_async_generator(sample_gen())
        print("Async result:", value)

    asyncio.run(test())
import json
import uuid
import traceback
from collections import defaultdict

# ---------------------------------------------------------
# Globals similar to JS
# ---------------------------------------------------------
NAMESPACE = None
CONTEXT = "window"
SELF_ID = uuid.uuid4().hex

pending_transactions = {}
handlers = {}

# ---------------------------------------------------------
# Error Serializer (JS → Python equivalent)
# ---------------------------------------------------------
def serialize_error(err):
    if isinstance(err, Exception):
        return {
            "name": err.__class__.__name__,
            "message": str(err),
            "stack": traceback.format_exc()
        }
    return str(err)

# ---------------------------------------------------------
# Safe Serializer
# ---------------------------------------------------------
def safe_serialize(obj, seen=None, depth=0, max_depth=20):
    if seen is None:
        seen = set()

    if depth >= max_depth:
        return "MaxDepthExceeded"

    oid = id(obj)
    if oid in seen:
        return "[Circular]"

    seen.add(oid)

    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, list):
        return [safe_serialize(x, seen, depth+1, max_depth) for x in obj]

    if isinstance(obj, dict):
        return {
            k: safe_serialize(v, seen, depth+1, max_depth)
            for k, v in obj.items()
        }

    # Generic object
    result = {}
    for key in dir(obj):
        if key.startswith("__"):
            continue
        try:
            value = getattr(obj, key)
            result[key] = safe_serialize(value, seen, depth+1, max_depth)
        except:
            pass

    return result

# ---------------------------------------------------------
# Register message handler
# ---------------------------------------------------------
def on_message(msg_id, handler):
    handlers[msg_id] = handler

# ---------------------------------------------------------
# Send message (simulating postMessage)
# ---------------------------------------------------------
def send_message(destination, message):
    # In a real system this would send via IPC, sockets, etc.
    # Here we just call the router directly.
    route_message(message)

# ---------------------------------------------------------
# Core router (JS function "b" + "w")
# ---------------------------------------------------------
def route_message(envelope):
    origin = envelope.get("origin", {})
    dest = envelope.get("destination")

    # Add hop
    hops = envelope.setdefault("hops", [])
    if SELF_ID in hops:
        return  # already passed here → prevent loop
    hops.append(SELF_ID)

    if dest and dest.get("context") == CONTEXT:
        # message for this context
        handle_incoming(envelope)
    else:
        # otherwise re-route
        send_message(dest, envelope)

# ---------------------------------------------------------
# Message handler (JS: function w)
# ---------------------------------------------------------
def handle_incoming(envelope):
    tx_id = envelope.get("transactionId")
    msg_id = envelope.get("messageID")
    msg_type = envelope.get("messageType")

    if msg_type == "reply":
        # resolve pending promise
        pending = pending_transactions.get(tx_id)
        if pending:
            if envelope.get("err"):
                pending["reject"](envelope["err"])
            else:
                pending["resolve"](envelope.get("data"))
            pending_transactions.pop(tx_id, None)
        return

    if msg_type == "message":
        handler = handlers.get(msg_id)
        if not handler:
            raise Exception(f"No handler for message id '{msg_id}'")

        try:
            result = handler({
                "sender": envelope.get("origin"),
                "id": msg_id,
                "data": envelope.get("data"),
                "timestamp": envelope.get("timestamp")
            })

            reply = {
                "transactionId": tx_id,
                "messageID": msg_id,
                "messageType": "reply",
                "data": result,
                "origin": {"context": CONTEXT},
                "destination": envelope.get("origin"),
                "hops": []
            }
            route_message(reply)

        except Exception as err:
            reply = {
                "transactionId": tx_id,
                "messageID": msg_id,
                "messageType": "reply",
                "err": serialize_error(err),
                "origin": {"context": CONTEXT},
                "destination": envelope.get("origin"),
                "hops": []
            }
            route_message(reply)

# ---------------------------------------------------------
# Sending request expecting reply
# ---------------------------------------------------------
def request(messageID, data, destination=None):
    tx_id = uuid.uuid4().hex

    envelope = {
        "transactionId": tx_id,
        "messageID": messageID,
        "messageType": "message",
        "data": data,
        "origin": {"context": CONTEXT},
        "destination": destination,
        "hops": []
    }

    # Create a Promise-like structure
    result = {}

    def resolve(value):
        result["value"] = value

    def reject(err):
        result["error"] = err

    pending_transactions[tx_id] = {"resolve": resolve, "reject": reject}

    route_message(envelope)

    if "error" in result:
        raise Exception(result["error"])
    return result.get("value")

import re
import uuid
import asyncio
import json
from enum import Enum, unique
from typing import Optional, Dict, Any, Callable

# --- regex for destination strings like "window@123.4" ---
DEST_RE = re.compile(r"^((?:background$)|devtools|popup|options|content-script|window)(?:@(\d+)(?:\.(\d+))?)?$")

def parse_destination(spec: str) -> Dict[str, Optional[int]]:
    """
    Parse destination spec string into a dict:
    e.g. "window@12.3" -> {"context":"window", "tabId":12, "frameId":3}
    """
    m = DEST_RE.match(spec or "")
    if not m:
        raise TypeError("Destination must be one of known destinations")
    context, tab, frame = m.group(1), m.group(2), m.group(3)
    return {
        "context": context,
        "tabId": int(tab) if tab else None,
        "frameId": int(frame) if frame else None
    }

# --- pending transactions map (transactionId -> future pair) ---
_pending: Dict[str, asyncio.Future] = {}

# route_message stub: replace with your router (b() / k() from earlier JS)
def route_message(envelope: Dict[str, Any]) -> None:
    """
    This should send the envelope to your routing layer.
    Here it's a placeholder — in your real app call the router (postMessage / IPC).
    """
    # Example: directly simulate an immediate reply by resolving the pending future.
    tx = envelope.get("transactionId")
    # This simulation simply echoes data back as result after a tiny delay:
    async def _simulate_reply():
        await asyncio.sleep(0.01)
        fut = _pending.pop(tx, None)
        if fut and not fut.done():
            # emulate reply object with .result same as JS "data"
            fut.set_result({"result": json.dumps({"echo": envelope.get("data")})})
    asyncio.get_event_loop().create_task(_simulate_reply())

# --- async send_message (JS: v) ---
async def send_message(message_id: str, data: Any, destination: str = "background") -> Any:
    dest = parse_destination(destination) if isinstance(destination, str) else destination
    if not dest.get("context"):
        raise TypeError("Destination must be any one of known destinations")

    transaction_id = uuid.uuid4().hex
    envelope = {
        "messageID": message_id,
        "data": data,
        "destination": dest,
        "messageType": "message",
        "transactionId": transaction_id,
        "origin": {"context": "window", "tabId": None},
        "hops": [],
        "timestamp": int(asyncio.get_event_loop().time() * 1000)
    }

    # create and store asyncio Future so caller can await reply
    loop = asyncio.get_event_loop()
    fut = loop.create_future()
    _pending[transaction_id] = fut

    # send (route) the message
    route_message(envelope)

    # wait for reply or error
    reply = await fut  # expected to be dict like {"result": "...", "error": "..."}

    # mimic the JS post-processing in O() - parse JSON or raise
    if reply is None:
        raise RuntimeError("No reply received")
    if "error" in reply and reply["error"]:
        raise Exception(json.loads(reply["error"]))
    if "result" in reply:
        return json.loads(reply["result"])
    return reply

# --- constants/enums from the JS snippet ---
@unique
class RequestTypes(Enum):
    WINDOW_REQUEST = "enkrypt_window_request"
    NEWWINDOW_REQUEST = "enkrypt_new_window_request"
    ACTION_REQUEST = "enkrypt_action_request"
    CS_REQUEST = "enkrypt_cs_request"
    BACKGROUND_REQUEST = "enkrypt_background_request"

@unique
class DestContexts(Enum):
    CONTENT_SCRIPT = "content-script"
    BACKGROUND = "background"
    WINDOW = "window"
    NEW_WINDOW = "new-window"
    POPUP = "popup"

@unique
class InjectPoint(Enum):
    MAIN = "enkrypt-inject"

@unique
class Methods(Enum):
    GET_ETH_ENCRYPTION_PUBKEY = "enkrypt_eth_encryption_pubkey"
    ETH_DECRYPT = "enkrypt_eth_decrypt"
    SIGN = "enkrypt_sign_hash"
    UNLOCK = "enkrypt_unlock_keyring"
    LOCK = "enkrypt_lock_keyring"
    IS_LOCKED = "enkrypt_is_locked_keyring"
    NEW_WINDOW_INIT = "enkrypt_newWindowInit"
    GET_SETTINGS = "enkrypt_getAllSettings"
    NEW_WINDOW_UNLOAD = "enkrypt_newWindowUnload"
    SEND_TO_TAB = "enkrypt_sendToTab"
    GET_NEW_ACCOUNT = "enkrypt_getNewAccount"
    SAVE_NEW_ACCOUNT = "enkrypt_saveNewAccount"
    CHANGE_NETWORK = "enkrypt_changeNetwork"

# --- convenience wrapper O (calls background and parses JSON or raises) ---
async def O(provider: str, message: Any) -> Any:
    raw = await send_message(RequestTypes.WINDOW_REQUEST.value, {"provider": provider, "message": message}, DestContexts.BACKGROUND.value)
    # raw expected structure { "result": "...", "error": "..." } per original code
    if raw.get("error"):
        raise Exception(json.loads(raw["error"]))
    return json.loads(raw["result"])

# --- small EventEmitter (JS emitter -> Python) ---
class EventEmitter:
    def __init__(self):
        self._events: Dict[str, list[Callable]] = {}

    def on(self, name: str, fn: Callable):
        self._events.setdefault(name, []).append(fn)
        return self

    def once(self, name: str, fn: Callable):
        def wrapper(*a, **k):
            self.off(name, wrapper)
            return fn(*a, **k)
        return self.on(name, wrapper)

    def off(self, name: str, fn: Optional[Callable] = None):
        if name not in self._events:
            return self
        if fn is None:
            del self._events[name]
        else:
            self._events[name] = [f for f in self._events[name] if f is not fn]
            if not self._events[name]:
                del self._events[name]
        return self

    def emit(self, name: str, *args, **kwargs):
        for fn in list(self._events.get(name, [])):
            fn(*args, **kwargs)
        return self
import enum

# -------------------------------------------------------------------
# EventEmitter (full translation of the JS prototype methods)
# -------------------------------------------------------------------

class EventEmitter:
    def __init__(self):
        self._events = {}      # eventName -> [handlers]
        self._eventsCount = 0

    def listenerCount(self, name):
        handlers = self._events.get(name)
        if not handlers:
            return 0
        return len(handlers)

    # emit with support for 0–many args (like JS switch-case)
    def emit(self, name, *args):
        if name not in self._events:
            return False

        handlers = list(self._events[name])   # copy to avoid mutation issues

        for h in handlers:
            fn = h["fn"]
            ctx = h["context"]
            once = h["once"]

            # call handler
            fn(*args) if ctx is None else fn.__get__(ctx)(*args)

            if once:
                self.removeListener(name, fn)

        return True

    # add normal listener
    def on(self, name, fn, context=None):
        self._events.setdefault(name, []).append({
            "fn": fn,
            "context": context,
            "once": False
        })
        self._eventsCount += 1
        return self

    addListener = on

    # add once-only listener
    def once(self, name, fn, context=None):
        self._events.setdefault(name, []).append({
            "fn": fn,
            "context": context,
            "once": True
        })
        self._eventsCount += 1
        return self

    # remove a specific listener
    def removeListener(self, name, fn=None):
        if name not in self._events:
            return self

        if fn is None:
            del self._events[name]
            self._eventsCount -= 1
            return self

        handlers = self._events[name]
        new_handlers = [h for h in handlers if h["fn"] != fn]

        if new_handlers:
            self._events[name] = new_handlers
        else:
            del self._events[name]
            self._eventsCount -= 1

        return self

    off = removeListener

    # remove all listeners for a given event
    def removeAllListeners(self, name=None):
        if name is None:
            self._events = {}
            self._eventsCount = 0
        else:
            if name in self._events:
                del self._events[name]
                self._eventsCount -= 1
        return self


# -------------------------------------------------------------------
# Enums (JS → Python)
# -------------------------------------------------------------------

class Namespace(enum.Enum):
    enkrypt = "enkrypt"
    ethereum = "ethereum"
    bitcoin = "bitcoin"
    polkadot = "polkadot"
    kadena = "kadena"
    solana = "solana"
    massa = "massa"


class StateKeys(enum.Enum):
    keyring = "KeyRing"
    persistentEvents = "PersistentEvents"
    domainState = "DomainState"
    evmAccountsState = "EVMAccountsState"
    substrateAccountsState = "SubstrateAccountsState"
    bitcoinAccountsState = "BitcoinAccountsState"
    kadenaAccountsState = "KadenaAccountsState"
    solanaAccountsState = "SolanaAccountsState"
    activityState = "ActivityState"
    marketData = "MarketData"
    cacheFetch = "CacheFetch"
    nftState = "NFTState"
    networksState = "NetworksState"
    settingsState = "SettingsState"
    tokensState = "TokensState"
    customNetworksState = "CustomNetworksState"
    rateState = "RateState"
    recentlySentAddresses = "RecentlySentAddresses"
    updatesState = "UpdatesState"
    backupState = "BackupState"
    menuState = "MenuState"
    bannersState = "BannersState"


class PersistentEvents(enum.Enum):
    persistentEvents = "PersistentEvents"
    chainChanged = "enkrypt_chainChanged"


class ChainType(enum.Enum):
    evm = 0
    substrate = 1
    bitcoin = 2
    kadena = 3
    solana = 4
    massa = 5


class EIP6963(enum.Enum):
    request = "eip6963:requestProvider"
    announce = "eip6963:announceProvider"


class EthNotifications(enum.Enum):
    changeChainId = "changeChainId"
    changeAddress = "changeAddress"
    changeConnected = "changeConnected"
    subscription = "eth_subscription"


class WalletEvents(enum.Enum):
    accountsChanged = "accountsChanged"
    chainChanged = "chainChanged"
    networkChanged = "networkChanged"
    connect = "connect"
    disconnect = "disconnect"
    message = "message"


class ErrorCodes(enum.Enum):
    userRejected = 4001
    unauthorized = 4100
    unsupportedMethod = 4200
    disconnected = 4900
    chainDisconnected = 4901


# Error definitions (just like JS `B`)
ERRORS = {
    4001: {
        "name": "User Rejected Request",
        "description": "The user rejected the request."
    },
    4100: {
        "name": "Unauthorized",
        "description": "The requested method/account is not authorized."
    },
    4200: {
        "name": "Unsupported Method",
        "description": "The provider does not support this method."
    },
    4900: {
        "name": "Disconnected",
        "description": "The provider is disconnected from all chains."
    },
    4901: {
        "name": "Chain Disconnected",
        "description": "Provider not connected to requested chain."
    }
}
import json
import uuid
import crypto       # replace with your actual crypto lib
from typing import Callable, Any, Dict

# Global subscription mapping
K: Dict[str, str] = {}

# --------------------------------------------------------------
# V – message handler function
# --------------------------------------------------------------
def handle_message_router(provider, raw_message: str):
    try:
        msg = json.loads(raw_message)
        method = msg.get("method")

        # ------------------------ changeConnected ------------------------
        if method == EthNotifications.changeConnected.value:
            is_connected = msg["params"][0]
            provider.connected = is_connected

            if is_connected:
                info = {"chainId": msg["params"][1]}

                if provider.chainId != info["chainId"]:
                    provider.chainId = info["chainId"]
                    provider.emit(WalletEvents.chainChanged.value, info["chainId"])

                provider.emit(WalletEvents.connect.value, info)

            else:
                error_code = msg["params"][1]
                if error_code not in ERRORS:
                    raise ValueError("Invalid error code")

                provider.emit(WalletEvents.disconnect.value, {
                    "code": error_code,
                    "message": f"{ERRORS[error_code]['name']}: {ERRORS[error_code]['description']}"
                })

        # ------------------------- changeChainId -------------------------
        elif method == EthNotifications.changeChainId.value:
            cid = msg["params"][0]
            if provider.chainId != cid:
                provider.chainId = cid
                provider.emit(WalletEvents.chainChanged.value, cid)

        # ------------------------- changeAddress -------------------------
        elif method == EthNotifications.changeAddress.value:
            addr = msg["params"][0]
            if provider.selectedAddress != addr:
                provider.selectedAddress = addr
                provider.emit(WalletEvents.accountsChanged.value, [addr])

        # -------------------------- subscription --------------------------
        elif method == EthNotifications.subscription.value:
            params = msg["params"]
            sub_id = params["subscription"]

            # map persistent subscription ID
            if sub_id in K:
                params["subscription"] = K[sub_id]

            provider.emit(WalletEvents.message.value, {
                "data": params,
                "type": method
            })

        # ---------------------- persistentEvents -------------------------
        elif method == PersistentEvents.persistentEvents.value:
            event_method = msg["params"][0]["method"]

            if event_method == "eth_subscribe":
                real_id = json.loads(msg["params"][1])
                temp_id = json.loads(msg["params"][2])
                K[temp_id] = real_id
            else:
                print(f"Unable to process persistentEvent: {raw_message}")

    except Exception as err:
        print("Message handling error:", err)


# --------------------------------------------------------------
# UUID generation (JS → Python)
# --------------------------------------------------------------

def generate_uuid():
    return str(uuid.uuid4())

# --------------------------------------------------------------
# q – The Enkrypt EVM provider (translated)
# --------------------------------------------------------------

class EnkryptProvider(EventEmitter):
    def __init__(self, config):
        super().__init__()
        self.chainId = None
        self.networkVersion = "0x1"
        self.isEnkrypt = True
        self.isMetaMask = True
        self.selectedAddress = None
        self.connected = True
        self.name = config["name"]
        self.type = config["type"]
        self.version = "2.13.1"
        self.autoRefreshOnNetworkChange = False
        self.sendMessageHandler: Callable = config["sendMessageHandler"]

    async def request(self, req: dict):
        # ensure chainId loaded
        if self.chainId is None:
            cid = await self.sendMessageHandler(self.name, json.dumps({
                "method": "eth_chainId"
            }))
            self.chainId = cid
            self.networkVersion = str(int(cid, 16))

        # request accounts
        if self.selectedAddress is None and req["method"] == "eth_requestAccounts":
            result = await self.sendMessageHandler(self.name, json.dumps(req))
            self.selectedAddress = result[0]
            return result

        return await self.sendMessageHandler(self.name, json.dumps(req))

    def enable(self):
        return self.request({"method": "eth_requestAccounts"})

    def isConnected(self):
        return self.connected

    def send(self, method_or_obj, params=None):
        if isinstance(method_or_obj, dict) and "method" in method_or_obj:
            return self.sendAsync(method_or_obj, params)
        else:
            return self.request({"method": method_or_obj, "params": params})

    def sendAsync(self, req, callback):
        method = req["method"]
        params = req.get("params", [])

        def _async():
            try:
                result = self.request({"method": method, "params": params})
                callback(None, {
                    "id": req.get("id"),
                    "jsonrpc": "2.0",
                    "result": result
                })
            except Exception as e:
                callback(e)

        return _async()

    def handleMessage(self, msg):
        handle_message_router(self, msg)


# --------------------------------------------------------------
# Proxy-like wrapper is replaced with Python-friendly injection
# --------------------------------------------------------------

class ProviderRegistry:
    def __init__(self):
        self.providers = {}

    def register(self, env, config):
        provider = EnkryptProvider(config)
        self.providers[config["name"]] = provider
        return provider

# -------------------------------
#  Accounts API
# -------------------------------

class Accounts:
    def __init__(self, opts):
        self.sendMessageHandler = opts["sendMessageHandler"]
        self.id = opts["id"]

    def get(self, include_metadata=False):
        return self.sendMessageHandler(self.id, {
            "method": "dot_accounts_get",
            "params": [bool(include_metadata)]
        })

    def subscribe(self, callback):
        def _unsubscribe():
            return None

        self.sendMessageHandler(self.id, {
            "method": "dot_accounts_get"
        }).then(lambda result: callback(result))

        return _unsubscribe


# -------------------------------
#  Metadata API
# -------------------------------

class Metadata:
    def __init__(self, opts):
        self.sendMessageHandler = opts["sendMessageHandler"]
        self.id = opts["id"]

    def get(self):
        return self.sendMessageHandler(self.id, {
            "method": "dot_metadata_get"
        })

    def provide(self, data):
        return self.sendMessageHandler(self.id, {
            "method": "dot_metadata_set",
            "params": [data]
        })


# -------------------------------
#  Provider (Not Implemented)
# -------------------------------

class Provider:
    def __init__(self, opts):
        import pyee
        self.eventEmitter = pyee.EventEmitter
        self.sendMessageHandler = opts["sendMessageHandler"]
        self.options = opts
        self.isClonable = True

    def clone(self):
        return Provider(self.options)

    async def connect(self):
        print("PostMessageProvider.connect() is not implemented.")

    async def disconnect(self):
        print("PostMessageProvider.disconnect() is not implemented.")

    @property
    def hasSubscriptions(self):
        print("PostMessageProvider.hasSubscriptions() is not implemented.")
        return False

    @property
    def isConnected(self):
        print("PostMessageProvider.isConnected() is not implemented.")
        return True

    async def listProviders(self):
        print("PostMessageProvider.listProviders() is not implemented.")
        return {}

    async def send(self, method, params, is_subscription, extra):
        print(method, params, "PostMessageProvider.send() is not implemented.")

    async def startProvider(self, providerId):
        print(providerId, "PostMessageProvider.startProvider() is not implemented.")
        return {
            "network": "",
            "node": "full",
            "source": "",
            "transport": ""
        }

    def on(self, event, callback):
        print("PostMessageProvider.on() is not implemented.")
        self.eventEmitter.on(event, callback)
        return lambda: None

    def subscribe(self, event, method, params, callback):
        print(event, method, params, "PostMessageProvider.subscribe() is not implemented.")
        return self.send(method, params, False, {
            "callback": callback,
            "type": event
        })

    async def unsubscribe(self, event, method, params):
        print("PostMessageProvider.unsubscribe() is not implemented.")
        return True


# -------------------------------
#  Signer API
# -------------------------------

_sign_counter = 0

class Signer:
    def __init__(self, opts):
        self.sendMessageHandler = opts["sendMessageHandler"]
        self.id = opts["id"]

    async def signPayload(self, payload):
        global _sign_counter
        result = await self.sendMessageHandler(self.id, {
            "method": "dot_signer_signPayload",
            "params": [payload]
        })
        _sign_counter += 1
        return {
            "signature": result,
            "id": _sign_counter
        }

    async def signRaw(self, payload):
        global _sign_counter
        result = await self.sendMessageHandler(self.id, {
            "method": "dot_signer_signRaw",
            "params": [payload]
        })
        _sign_counter += 1
        return {
            "signature": result,
            "id": _sign_counter
        }


# -------------------------------
#  Main Wrapper API (re)
# -------------------------------

class EnkryptDOT:
    def __init__(self, opts):
        self.dappName = opts["dappName"]
        self.id = opts["id"]
        self.sendMessageHandler = opts["sendMessageHandler"]

        # Proxies removed → direct instances
        self.accounts = Accounts(opts)
        self.metadata = Metadata(opts)
        self.provider = Provider(opts)
        self.signer = Signer(opts)

    def handleMessage(self, msg):
        method = msg.get("method")
        params = msg.get("params")
        print(method, params)


# -------------------------------
#  Provider Manager (ie)
# -------------------------------

class ProviderManager:
    def __init__(self):
        self.providers = []

    def addProvider(self, provider):
        self.providers.append(provider)

    def nextPosition(self):
        return len(self.providers)

    def handleMessage(self, raw_msg):
        msg = json.loads(raw_msg)
        provider_id = msg.get("id")

        if provider_id < len(self.providers):
            self.providers[provider_id].handleMessage({
                "method": msg.get("method"),
                "params": msg.get("params")
            })
        else:
            print(f"Provider id missing: {raw_msg} id: {provider_id}")


# Instantiate Manager (same as JS "ie")
import json
ie = ProviderManager()
class EventEmitter:
    def __init__(self):
        self.listeners = {}

    def on(self, event, callback):
        self.listeners.setdefault(event, []).append(callback)

    def emit(self, event, *args):
        for cb in self.listeners.get(event, []):
            cb(*args)
class CE(EventEmitter):
    version = "2.13.1"

    def __init__(self, opts):
        super().__init__()
        self.name = opts["name"]
        self.type = opts["type"]
        self.sendMessageHandler = opts["sendMessageHandler"]

        # Global ae function recreated inside Python
        global ae
        def ae(t, n):
            method = n["method"]
            params = n.get("params")
            message = {
                "id": t,
                "method": method,
                "params": params
            }
            return opts["sendMessageHandler"](opts["name"], json.dumps(message))
        ae = ae

    def handleMessage(self, message):
        IE.handleMessage(message)

    def enable(self, dapp_name):
        position = IE.nextPosition()
        provider = RE({
            "dappName": dapp_name,
            "sendMessageHandler": ae,
            "id": position
        })
        IE.addProvider(provider)
        return provider
class ProxyHandler:
    proxymethods = ["enable"]

    @classmethod
    def ownKeys(cls, obj):
        return list(obj.__dict__.keys()) + cls.proxymethods

    @classmethod
    def has(cls, obj, attr):
        return attr in cls.ownKeys(obj)
from enum import Enum

class UE(Enum):
    Ethereum = "ETH"
    Okc = "OKT"
    Binance = "BNB"
    EthereumClassic = "ETC"
    Sepolia = "SEPOLIA"
    Matic = "MATIC"
    MaticZK = "MATICZK"
    Moonbeam = "GLMR"
    Moonriver = "MOVR"
    Rootstock = "Rootstock"
    RootstockTestnet = "RootstockTestnet"
    Acala = "ACA"
    Karura = "KAR"
    KaruraEVM = "evmKAR"
    Kusama = "KSM"
    Polkadot = "DOT"
    Westend = "WND"
    Bitcoin = "BTC"
    BitcoinTest = "BTCTest"
    Astar = "ASTR"
    Shiden = "SDN"
    ShidenEVM = "SDNEVM"
    AstarEVM = "ASTREVM"
    Optimism = "OP"
    Canto = "CANTO"
    # ... (all remaining values exactly same)
class LE(Enum):
    Ethereum = "ethereum"
    Binance = "binance-smart-chain"
    EthereumClassic = "ethereum-classic"
    Matic = "polygon-pos"
    MaticZK = "polygon-zkevm"
    Moonbeam = "moonbeam"
    Moonriver = "moonriver"
    Acala = "acala"
    Karura = "karura"
    KaruraEVM = "karura"
    Kusama = "kusama"
    Polkadot = "polkadot"
    Rootstock = "rootstock"
    Okc = "okex-chain"
    # ... continue others same as JS list
class HE(Enum):
    mnemonic = "mnemonic"
    privkey = "privkey"
    ledger = "ledger"
    trezor = "trezor"
class PE(Enum):
    ledger = "ledger"
    trezor = "trezor"
class YE(Enum):
    UnableToVerify = "Signing verification failed"
    NotSupported = "Sign type not supported"
class PE(Enum):
    ledger = "ledger"
    trezor = "trezor"
import json

class EventEmitter:
    def __init__(self):
        self.listeners = {}

    def on(self, event, callback):
        self.listeners.setdefault(event, []).append(callback)

    def emit(self, event, *args):
        for cb in self.listeners.get(event, []):
            cb(*args)
we = {
    "livenet": UE.Bitcoin,
    "testnet": UE.BitcoinTest,
    "litecoin": UE.Litecoin
}
class BE(EventEmitter):
    version = "2.13.1"
    autoRefreshOnNetworkChange = False

    def __init__(self, opts):
        super().__init__()
        self.connected = True
        self.name = opts["name"]
        self.type = opts["type"]
        self.networks = we
        self.sendMessageHandler = opts["sendMessageHandler"]

    async def request(self, payload):
        return await self.sendMessageHandler(self.name, json.dumps(payload))

    async def requestAccounts(self):
        return await self.request({"method": "btc_requestAccounts"})

    async def getAccounts(self):
        return await self.request({"method": "btc_requestAccounts"})

    async def getPublicKey(self):
        return await self.request({"method": "btc_getPublicKey"})

    async def getNetwork(self):
        return await self.request({"method": "btc_getNetwork"})

    async def switchNetwork(self, net):
        return await self.request({
            "method": "btc_switchNetwork",
            "params": [net]
        })

    async def getBalance(self):
        return await self.request({"method": "btc_getBalance"})

    async def signPsbt(self, psbt, options):
        return await self.request({
            "method": "btc_signPsbt",
            "params": [psbt, options]
        })

    async def signMessage(self, msg, options):
        return await self.request({
            "method": "btc_signMessage",
            "params": [msg, options]
        })

    # --- Not implemented methods ---
    async def getInscriptions(self):
        raise Exception("not implemented")

    async def sendBitcoin(self):
        raise Exception("not implemented")

    async def sendInscription(self):
        raise Exception("not implemented")

    async def inscribeTransfer(self):
        raise Exception("not implemented")

    async def pushTx(self):
        raise Exception("not implemented")

    async def signPsbts(self):
        raise Exception("not implemented")

    async def pushPsbt(self):
        raise Exception("not implemented")

    def isConnected(self):
        return self.connected
import json

# ----------------- EventEmitter helper -----------------
class EventEmitter:
    def __init__(self):
        self.events = {}

    def on(self, event, handler):
        self.events.setdefault(event, []).append(handler)

    def emit(self, event, *args):
        if event in self.events:
            for handler in self.events[event]:
                handler(*args)


# ----------------- Constants -----------------
class _:
    changeConnected = "changeConnected"
    changeAddress = "changeAddress"
    chainChanged = "chainChanged"


class U:
    connect = "connect"
    disconnect = "disconnect"
    accountsChanged = "accountsChanged"
    networkChanged = "networkChanged"


class ue:
    Bitcoin = "bitcoin"
    BitcoinTest = "bitcoin-test"
    Kadena = "kadena"
    KadenaTestnet = "kadena-testnet"


# ----------------- Network Mapping -----------------
Ne = {
    "KDA": ue.Kadena,
    "KDATestnet": ue.KadenaTestnet
}

# ----------------- ke / je enums -----------------
class ke:
    changeAddress = "changeAddress"
    changeNetwork = "changeNetwork"


class je:
    accountsChanged = "accountsChanged"
    networkChanged = "networkChanged"


# ----------------- Main Class ve -----------------
class R(EventEmitter):
    pass


class Ve(R):
    version = "2.13.1"

    def __init__(self, config):
        super().__init__()
        self.connected = True
        self.name = config["name"]
        self.type = config["type"]
        self.networks = Ne
        self.autoRefreshOnNetworkChange = False
        self.sendMessageHandler = config["sendMessageHandler"]

    async def request(self, payload):
        return await self.sendMessageHandler(self.name, json.dumps(payload))

    def isConnected(self):
        return self.connected

    def handleMessage(self, raw_msg):
        try:
            data = json.loads(raw_msg)

            if data["method"] == ke.changeAddress:
                addr = data["params"][0]
                self.emit(je.accountsChanged, [addr])

            elif data["method"] == ke.changeNetwork:
                net = data["params"][0]
                self.emit(je.networkChanged, [net])

            else:
                print(f"Unable to process message: {raw_msg}")

        except Exception as err:
            print("Error:", err)


# ----------------- Wallet Register Event -----------------
class Ie:
    def __init__(self, detail):
        self.detail = detail
        self.type = "wallet-standard:register-wallet"

    def preventDefault(self):
        raise Exception("preventDefault cannot be called")

    def stopImmediatePropagation(self):
        raise Exception("stopImmediatePropagation cannot be called")

    def stopPropagation(self):
        raise Exception("stopPropagation cannot be called")


# ----------------- Solana Features -----------------
Ae = "solana:signAndSendTransaction"
Se = "solana:signIn"
De = "solana:signMessage"
Oe = "solana:signTransaction"
Te = "standard:connect"
ze = "standard:disconnect"
Ee = "standard:events"

xe = ["solana:mainnet"]


def Ce(chain):
    return chain in xe


Pe = xe
Le = [Ae, Oe, De]


# ----------------- Wallet Class -----------------
class _e:
    def __init__(self, address, publicKey, label, icon):
        self.address = address
        self.publicKey = publicKey[:]
        self.chains = Pe[:]
        self.features = Le[:]
        self.label = label
        self.icon = icon
import json

# ----------------- EventEmitter helper -----------------
class EventEmitter:
    def __init__(self):
        self.events = {}

    def on(self, event, handler):
        self.events.setdefault(event, []).append(handler)

    def emit(self, event, *args):
        if event in self.events:
            for handler in self.events[event]:
                handler(*args)


# ----------------- Constants -----------------
class _:
    changeConnected = "changeConnected"
    changeAddress = "changeAddress"
    chainChanged = "chainChanged"


class U:
    connect = "connect"
    disconnect = "disconnect"
    accountsChanged = "accountsChanged"
    networkChanged = "networkChanged"


class ue:
    Bitcoin = "bitcoin"
    BitcoinTest = "bitcoin-test"
    Kadena = "kadena"
    KadenaTestnet = "kadena-testnet"


# ----------------- Network Mapping -----------------
Ne = {
    "KDA": ue.Kadena,
    "KDATestnet": ue.KadenaTestnet
}

# ----------------- ke / je enums -----------------
class ke:
    changeAddress = "changeAddress"
    changeNetwork = "changeNetwork"


class je:
    accountsChanged = "accountsChanged"
    networkChanged = "networkChanged"


# ----------------- Main Class ve -----------------
class R(EventEmitter):
    pass


class Ve(R):
    version = "2.13.1"

    def __init__(self, config):
        super().__init__()
        self.connected = True
        self.name = config["name"]
        self.type = config["type"]
        self.networks = Ne
        self.autoRefreshOnNetworkChange = False
        self.sendMessageHandler = config["sendMessageHandler"]

    async def request(self, payload):
        return await self.sendMessageHandler(self.name, json.dumps(payload))

    def isConnected(self):
        return self.connected

    def handleMessage(self, raw_msg):
        try:
            data = json.loads(raw_msg)

            if data["method"] == ke.changeAddress:
                addr = data["params"][0]
                self.emit(je.accountsChanged, [addr])

            elif data["method"] == ke.changeNetwork:
                net = data["params"][0]
                self.emit(je.networkChanged, [net])

            else:
                print(f"Unable to process message: {raw_msg}")

        except Exception as err:
            print("Error:", err)


# ----------------- Wallet Register Event -----------------
class Ie:
    def __init__(self, detail):
        self.detail = detail
        self.type = "wallet-standard:register-wallet"

    def preventDefault(self):
        raise Exception("preventDefault cannot be called")

    def stopImmediatePropagation(self):
        raise Exception("stopImmediatePropagation cannot be called")

    def stopPropagation(self):
        raise Exception("stopPropagation cannot be called")


# ----------------- Solana Features -----------------
Ae = "solana:signAndSendTransaction"
Se = "solana:signIn"
De = "solana:signMessage"
Oe = "solana:signTransaction"
Te = "standard:connect"
ze = "standard:disconnect"
Ee = "standard:events"

xe = ["solana:mainnet"]


def Ce(chain):
    return chain in xe


Pe = xe
Le = [Ae, Oe, De]


# ----------------- Wallet Class -----------------
class _e:
    def __init__(self, address, publicKey, label, icon):
        self.address = address
        self.publicKey = publicKey[:]
        self.chains = Pe[:]
        self.features = Le[:]
        self.label = label
        self.icon = icon
from typing import List, Callable, Dict, Optional

# --- Utility functions ---
def hex_to_bytes(hex_str: str) -> bytes:
    """Convert hex string (with or without 0x) to bytes."""
    hex_str = hex_str.replace("0x", "")
    return bytes.fromhex(hex_str)

def bytes_to_hex(data: bytes) -> str:
    """Convert bytes back to hex string with 0x prefix."""
    return "0x" + data.hex()


# --- Account representation ---
class Account:
    def __init__(self, address: str, public_key: bytes):
        self.address = address
        self.public_key = public_key

    def __repr__(self):
        return f"Account(address={self.address}, public_key={bytes_to_hex(self.public_key)})"


# --- Wallet Adapter ---
class EnkryptAdapter:
    def __init__(self, provider):
        self._version = "1.0.0"
        self._name = "Enkrypt"
        self._icon = "data:image/svg+xml;base64,..."
        self._provider = provider
        self._accounts: Optional[List[Account]] = None
        self._listeners: Dict[str, List[Callable]] = {}

        # Hook provider events
        provider.on("connect", self._on_connect)
        provider.on("disconnect", self._on_disconnect)
        provider.on("accountsChanged", self._on_accounts_changed)

        # Initialize
        self._on_connect()

    # --- Properties ---
    @property
    def version(self) -> str:
        return self._version

    @property
    def name(self) -> str:
        return self._name

    @property
    def icon(self) -> str:
        return self._icon

    @property
    def accounts(self) -> List[Account]:
        return self._accounts or []

    # --- Event system ---
    def on(self, event: str, callback: Callable):
        self._listeners.setdefault(event, []).append(callback)
        return lambda: self._remove_listener(event, callback)

    def _emit(self, event: str, *args, **kwargs):
        for cb in self._listeners.get(event, []):
            cb(*args, **kwargs)

    def _remove_listener(self, event: str, callback: Callable):
        if event in self._listeners:
            self._listeners[event] = [cb for cb in self._listeners[event] if cb != callback]

    # --- Provider hooks ---
    def _on_connect(self, *args, **kwargs):
        if self._provider.accounts:
            self._accounts = [
                Account(acc["address"], hex_to_bytes(acc["pubkey"]))
                for acc in self._provider.accounts
            ]
            self._emit("change", {"accounts": self.accounts})

    def _on_disconnect(self, *args, **kwargs):
        if self._accounts:
            self._accounts = None
            self._emit("change", {"accounts": []})

    def _on_accounts_changed(self, *args, **kwargs):
        if self._provider.accounts:
            self._on_connect()
        else:
            self._on_disconnect()

    # --- Public methods ---
    async def connect(self, silent: bool = False):
        if not self._accounts:
            await self._provider.connect({"onlyIfTrusted": True} if silent else None)
        self._on_connect()
        return {"accounts": self.accounts}

    async def disconnect(self):
        await self._provider.disconnect()
        self._on_disconnect()


class EnkryptAdapter:
    # ... (previous code from earlier message)

    async def sign_and_send_transaction(self, *requests):
        """
        Equivalent to #y in JS.
        Each request should be a dict with keys:
        { "transaction": bytes, "account": Account, "chain": str, "options": dict }
        """
        if not self._accounts:
            raise RuntimeError("not connected")

        results = []
        for req in requests:
            tx = req["transaction"]
            account = req["account"]
            chain = req["chain"]
            options = req.get("options", {})

            # Validate account
            if not any(acc.address == account.address for acc in self._accounts):
                raise RuntimeError("invalid account")

            # Validate chain (Ce equivalent: placeholder check)
            if not self._validate_chain(chain):
                raise RuntimeError("invalid chain")

            # Call provider
            sig = await self._provider.signAndSendTransaction({
                "address": account.address,
                "hex": bytes_to_hex(tx),
                "chain": chain
            }, options)

            results.append({"signature": hex_to_bytes(sig)})
        return results

    async def sign_transaction(self, *requests):
        """
        Equivalent to #m in JS.
        Each request: { "transaction": bytes, "account": Account, "chain": str }
        """
        if not self._accounts:
            raise RuntimeError("not connected")

        results = []
        for req in requests:
            tx = req["transaction"]
            account = req["account"]
            chain = req.get("chain")

            if not any(acc.address == account.address for acc in self._accounts):
                raise RuntimeError("invalid account")

            if chain and not self._validate_chain(chain):
                raise RuntimeError("invalid chain")

            signed_tx = await self._provider.signTransaction({
                "address": account.address,
                "hex": bytes_to_hex(tx),
                "chain": chain
            })

            results.append({"signedTransaction": hex_to_bytes(signed_tx)})
        return results

    async def sign_message(self, *requests):
        """
        Equivalent to #M in JS.
        Each request: { "message": bytes, "account": Account }
        """
        if not self._accounts:
            raise RuntimeError("not connected")

        results = []
        if len(requests) == 1:
            req = requests[0]
            message = req["message"]
            account = req["account"]

            if not any(acc.address == account.address for acc in self._accounts):
                raise RuntimeError("invalid account")

            resp = await self._provider.signMessage({
                "address": account.address,
                "message": bytes_to_hex(message)
            })

            results.append({
                "signedMessage": hex_to_bytes(resp["signedMessage"]),
                "signature": hex_to_bytes(resp["signature"])
            })
        else:
            for req in requests:
                results.extend(await self.sign_message(req))
        return results

    async def sign_in(self, *requests):
        """
        Equivalent to #f in JS.
        Each request is passed directly to provider.signIn.
        """
        results = []
        for req in requests:
            resp = await self._provider.signIn(req)
            account = Account(
                address=resp["address"],
                public_key=hex_to_bytes(resp["pubkey"])
            )
            results.append({
                "account": account,
                "signature": hex_to_bytes(resp["signature"]),
                "signedMessage": hex_to_bytes(resp["signedMessage"]),
                "signatureType": resp["signatureType"]
            })

        # Refresh accounts after sign-in
        self._on_connect()
        return results

    # --- Helper ---
    def _validate_chain(self, chain: str) -> bool:
        """Placeholder for Ce(chain) check in JS."""
        # In JS, Ce(r) validates chain. Here you can implement your own logic.
        return isinstance(chain, str) and len(chain) > 0



import json
from typing import Any, Dict, List, Optional

class SolanaProvider:
    def __init__(self, name: str, send_message_handler):
        self.name = name
        self.type = "solana"
        self.send_message_handler = send_message_handler
        self.connected: bool = True
        self.accounts: List[Dict[str, Any]] = []

    async def connect(self, options: Optional[Dict] = None):
        """Equivalent to connect(e) in JS."""
        req = {
            "method": "sol_connect",
            "params": [options] if options else []
        }
        result = await self.request(req)
        self.accounts = result
        return result

    async def disconnect(self):
        """Equivalent to disconnect() in JS."""
        self.accounts = []
        self.connected = False
        return None

    async def sign_and_send_transaction(self, tx: Dict, opts: Dict):
        """Equivalent to signAndSendTransaction(e, t)."""
        req = {
            "method": "sol_signAndSendTransaction",
            "params": [json.dumps(tx), json.dumps(opts)]
        }
        return await self.request(req)

    async def sign_transaction(self, tx: Dict):
        """Equivalent to signTransaction(e)."""
        req = {
            "method": "sol_signTransaction",
            "params": [json.dumps(tx)]
        }
        return await self.request(req)

    async def sign_message(self, msg: Dict):
        """Equivalent to signMessage(e)."""
        req = {
            "method": "sol_signMessage",
            "params": [json.dumps(msg)]
        }
        return await self.request(req)

    async def sign_in(self, payload: Dict):
        """Equivalent to signIn(e)."""
        req = {
            "method": "sol_signInMessage",
            "params": [json.dumps(payload)]
        }
        result = await self.request(req)

        # Update accounts list if new
        if not any(acc["address"] == result["address"] for acc in self.accounts):
            self.accounts.append({
                "address": result["address"],
                "pubkey": result["pubkey"]
            })
        return result

    async def request(self, payload: Dict):
        """Equivalent to request(e)."""
        return await self.send_message_handler(self.name, json.dumps(payload))

    def is_connected(self) -> bool:
        return self.connected

    def handle_message(self, raw_message: str):
        """Equivalent to handleMessage(e)."""
        try:
            msg = json.loads(raw_message)
            if msg["method"] == "changeConnected":
                self.connected = msg["params"][0]
                # In JS they emit events here; in Python you could call callbacks
            elif msg["method"] == "changeAddress":
                # Refresh accounts
                # Equivalent to calling sol_connect again
                # Here we just simulate
                pass
        except Exception as err:
            print("Error handling message:", err)


#provider = SolanaProvider("solana", send_message_handler)
#adapter = EnkryptAdapter(provider)

#await adapter.connect()
#await adapter.sign_transaction({"tx": "example"})

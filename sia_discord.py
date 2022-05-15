import discord
import discord.ext.commands as cmd
from discord.ext.commands import Context
import json

import sia_vm
import siamodel
from pprint import pprint
    
class SIADiscordBot(cmd.Cog):
    def __init__(self, bot, vm_manager, vm_sessions, responder, config_json="./config/bot-config.json"):
        self.bot = cmd
    
        f = open(config_json, "r")
        config = json.load(f)
        f.close()
        self.prefix  = config['Prefix']
        self.admin  = config['Admin']
        self.verbose = config["verbose"]
        
        self.vm_manager = vm_manager
        self.vm_sessions = vm_sessions
        self.responder = responder 
        
        with open("./commands/help.txt", "r") as f:
            self.detailed_help = f.read()
        super().__init__()


    @cmd.command()
    async def connect(self, ctx, *message):
        client_id = ctx.message.author.id
        self.sia_connect(client_id)
        await ctx.send("A new vm session launched")

    @cmd.command()
    async def eval(self, ctx, *message):
        client_id = ctx.message.author.id
        if client_id in self.admin:
            session = self.sia_connect(client_id)
            if len(message) < 1:
                await ctx.send("Check your input, {} eval <code>".format(self.prefix))
            else:
                self.log("{} is trying to evaluate {}".format(client_id, message[0]))
                session.set_code(self.vm_manager, message[0])
                session.parse_args(self.vm_manager, "Testに")
                await session.vm_run(ctx, self.vm_manager)
                await ctx.send("Your order succesfully done.")
        else:
            await ctx.send("I'm sorry but this command is only for admins, please contact to owners.")
    
    @cmd.command()
    async def ask(self, ctx, *message):
        client_id = ctx.message.author.id
        if client_id in self.admin:
            input_txt = " ".join(message)
            vmcode = self.responder.predict_sentence(input_txt)
            session = self.sia_connect(client_id)
            session.parse_args(self.vm_manager, input_txt)
            pprint(self.vm_manager.interpret(vmcode))
            session.set_vm(self.vm_manager, vmcode)
            await session.vm_run(ctx, self.vm_manager)
        else:
            await ctx.send("I'm sorry but this command is only for admins because this bot is under maintenance.")
    
    @cmd.command()
    async def help(self, ctx, *message):
        await ctx.send(self.detailed_help)
        
    @cmd.command()
    async def decode(self, ctx, *message):
        """
        For debug
        """
        print(self.vm_manager.sia_dict.sia_encode)
        await ctx.send(self.vm_manager.decode_str([int(message[0])]))
        
    def sia_connect(self, client_id):
        if client_id in self.vm_sessions.keys():
            return self.vm_sessions[client_id]
        else:
            self.log("New Session started for {}".format(client_id))
            self.vm_sessions[client_id] = sia_vm.SIAVMSession(client_id)
            return self.vm_sessions[client_id]
    
    def log(self, content):
        if self.verbose:
            print("[LOG]: {}".format(content))
    
        
class SIA(cmd.Bot):
    def __init__(self, config_json="./config/bot-config.json"):
        f = open(config_json, "r")
        config = json.load(f)
        
        self.token  = config['Token']
        self.prefix = config['Prefix']
        self.admin  = config['Admin']
        self.verbose = config["verbose"]
        
        self.vm_manager = sia_vm.SIABinaryManager(self)
        self.vm_sessions = {}
        self.responder = siamodel.TransformerResponder(self.vm_manager)
        
        super().__init__(command_prefix=self.prefix, help_command=None)
        self.add_cog(SIADiscordBot(self, self.vm_manager, self.vm_sessions, self.responder))
        
    async def on_ready(self):
        game = discord.Game("sia ask どうやって使うの？".format(self.prefix))
        await self.change_presence(status=discord.Status.online, activity=game)
    
    async def on_message(self, message):
        if message.channel.id in [973087985551683614, 974061433455259678] and message.author.id in self.admin:
            session = self.sia_connect(message.author.id)
            vmcode = self.responder.predict_sentence(message.content)
            session.parse_args(self.vm_manager, message.content)
            pprint(self.vm_manager.interpret(vmcode))
            session.set_vm(self.vm_manager, vmcode)
            ctx = Context(message=message,bot=self,prefix=self.prefix)
            await session.vm_run(ctx, self.vm_manager)
        await self.process_commands(message)
        
    def run_with_token(self):
        self.run(self.token)

    def sia_connect(self, client_id):
        if client_id in self.vm_sessions.keys():
            return self.vm_sessions[client_id]
        else:
            self.log("New Session started for {}".format(client_id))
            self.vm_sessions[client_id] = sia_vm.SIAVMSession(client_id)
            return self.vm_sessions[client_id]
    
    def log(self, content):
        if self.verbose:
            print("[LOG]: {}".format(content))
            
    
sia_bot = SIA()
print("SIA Started===")
sia_bot.run_with_token()

s = SIAVMSession()
s.set_code(manager, "daily")

s.vm_run(manager)

s.display_i()


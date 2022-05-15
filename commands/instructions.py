import sia_vm

class reply(sia_vm.SIAInstruction):
    async def execute(self, session, ctx, manager, args):
        """
        Reply User
        """
        content = session.stack_pop()
        await ctx.send(content)
        
class reply_dm(sia_vm.SIAInstruction):
    async def execute(self, session, ctx, manager, args):
        """
        Reply User
        """
        content = session.stack_pop()
        await ctx.author.send(content)
    
class push(sia_vm.SIAInstruction):
    async def execute(self, session, ctx, manager, args):
        """
        push stack
        """
        result = []
        for s in args:
            if type(s) == int:
                result.append(manager.decode_str([s]))
            else:
                result.append(self.get_pronouns(session, ctx, s))
        session.stack_push("".join(result))
    
class pushl(sia_vm.SIAInstruction):
    async def execute(self, session, ctx, manager, args):
        """
        push urls to stack
        """
        obj = manager.decode_str(args[1:][:-1]) + "/"
        session.stack_push(obj)
    
class merge(sia_vm.SIAInstruction):
    async def execute(self, session, ctx, manager, args):
        """
        concatenate two strings from stack
        """
        obj1 = session.stack_pop()
        obj2 = session.stack_pop()
        session.stack_push(obj1 + obj2)
    
class send(sia_vm.SIAInstruction):
    async def execute(self, session, ctx, manager, args):
        if args[0] == "analyze_args":
            session.args, session.obj, session.obl = manager.analyzer.parse(session.user_last_input)
        elif args[0] == "obj":
            session.stack_push(session.obj)
        elif args[0] == "obl":
            session.stack_push(session.obl)
        elif args[0] == "search":
            selected_engine = session.stack_pop()[0].getall(exclude_pos=True)[0]
            selected_word   = session.stack_pop()[0].getall(exclude_pos=True)[0]
            session.stack_push(selected_engine + "での検索結果:" + selected_word)
        elif args[0] == "register_memo":
            obj = session.stack_pop()
            title = ""
            content = "".join(session.stack_pop()[0].getall())[:-1] #を<-こいつを排除
            
            for obj_ in obj:
                if obj_.get("Time"):
                    title = obj_.get("Time")
                    
            if title == "":
                title = "".join(obj[0].getall(exclude_pos=True)[:-1])
                
            session.user_info.write_dict("memo", title, content)
            session.stack_push(title)
            session.stack_push("予定をTitle:\"" +title + "\"として登録しました。")
        elif args[0] == "delete_memo":
            obj = session.stack_pop()
            title = ""
            for obj_ in obj:
                if obj_.get("Time"):
                    title = obj_.get("Time")
                    
            if title == "":
                title = "".join(obj[0].getall(exclude_pos=True)[:-1])
            session.user_info.delete_dict("memo", title)
            session.stack_push(title + "を削除しました")
            
        elif args[0] == "find_memos":
            obj = session.stack_pop()
            title = ""
            for obj_ in obj:
                if obj_.get("Time"):
                    title = obj_.get("Time")
                    
            if title == "":
                title = "".join(obj[0].getall(exclude_pos=True)[:-1])
            session.stack_push(session.user_info.get_memos(title))

                               
class input(sia_vm.SIAInstruction):
    async def execute(self, session, ctx, manager, args):
        def check(msg):
            return msg.author == ctx.author and msg.channel == ctx.channel
        msg = session.stack_pop()
        await ctx.send(msg)
        await ctx.channel.trigger_typing()
        try:
            msg = await manager.bot.wait_for("message", check=check, timeout=8)
            msg = msg.content
        except:
            await ctx.send("遅いよぅ。。。時間切れ！")
            msg = "nothing"
            
        session.stack_push(msg)
    
class call_talk_api(sia_vm.SIAInstruction):
    async def execute(self, session, ctx, manager, args):
        pass
"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[711],{6148:(e,n,t)=>{t.r(n),t.d(n,{assets:()=>c,contentTitle:()=>l,default:()=>h,frontMatter:()=>a,metadata:()=>o,toc:()=>r});var i=t(4848),s=t(8453);const a={sidebar_label:"action",title:"opendevin.schema.action"},l=void 0,o={id:"python/opendevin/schema/action",title:"opendevin.schema.action",description:"ActionTypeSchema Objects",source:"@site/docs/python/opendevin/schema/action.md",sourceDirName:"python/opendevin/schema",slug:"/python/opendevin/schema/action",permalink:"/docs/python/opendevin/schema/action",draft:!1,unlisted:!1,tags:[],version:"current",frontMatter:{sidebar_label:"action",title:"opendevin.schema.action"},sidebar:"apiSidebar",previous:{title:"requirement",permalink:"/docs/python/opendevin/sandbox/plugins/requirement"},next:{title:"observation",permalink:"/docs/python/opendevin/schema/observation"}},c={},r=[{value:"ActionTypeSchema Objects",id:"actiontypeschema-objects",level:2},{value:"INIT",id:"init",level:4},{value:"START",id:"start",level:4},{value:"READ",id:"read",level:4},{value:"WRITE",id:"write",level:4},{value:"RUN",id:"run",level:4},{value:"KILL",id:"kill",level:4},{value:"BROWSE",id:"browse",level:4},{value:"RECALL",id:"recall",level:4},{value:"THINK",id:"think",level:4},{value:"DELEGATE",id:"delegate",level:4},{value:"FINISH",id:"finish",level:4},{value:"PAUSE",id:"pause",level:4},{value:"RESUME",id:"resume",level:4},{value:"STOP",id:"stop",level:4}];function d(e){const n={code:"code",h2:"h2",h4:"h4",p:"p",pre:"pre",...(0,s.R)(),...e.components};return(0,i.jsxs)(i.Fragment,{children:[(0,i.jsx)(n.h2,{id:"actiontypeschema-objects",children:"ActionTypeSchema Objects"}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"class ActionTypeSchema(BaseModel)\n"})}),"\n",(0,i.jsx)(n.h4,{id:"init",children:"INIT"}),"\n",(0,i.jsx)(n.p,{children:"Initializes the agent. Only sent by client."}),"\n",(0,i.jsx)(n.h4,{id:"start",children:"START"}),"\n",(0,i.jsx)(n.p,{children:"Starts a new development task. Only sent by the client."}),"\n",(0,i.jsx)(n.h4,{id:"read",children:"READ"}),"\n",(0,i.jsx)(n.p,{children:"Reads the content of a file."}),"\n",(0,i.jsx)(n.h4,{id:"write",children:"WRITE"}),"\n",(0,i.jsx)(n.p,{children:"Writes the content to a file."}),"\n",(0,i.jsx)(n.h4,{id:"run",children:"RUN"}),"\n",(0,i.jsx)(n.p,{children:"Runs a command."}),"\n",(0,i.jsx)(n.h4,{id:"kill",children:"KILL"}),"\n",(0,i.jsx)(n.p,{children:"Kills a background command."}),"\n",(0,i.jsx)(n.h4,{id:"browse",children:"BROWSE"}),"\n",(0,i.jsx)(n.p,{children:"Opens a web page."}),"\n",(0,i.jsx)(n.h4,{id:"recall",children:"RECALL"}),"\n",(0,i.jsx)(n.p,{children:"Searches long-term memory"}),"\n",(0,i.jsx)(n.h4,{id:"think",children:"THINK"}),"\n",(0,i.jsx)(n.p,{children:"Allows the agent to make a plan, set a goal, or record thoughts"}),"\n",(0,i.jsx)(n.h4,{id:"delegate",children:"DELEGATE"}),"\n",(0,i.jsx)(n.p,{children:"Delegates a task to another agent."}),"\n",(0,i.jsx)(n.h4,{id:"finish",children:"FINISH"}),"\n",(0,i.jsx)(n.p,{children:"If you're absolutely certain that you've completed your task and have tested your work,\nuse the finish action to stop working."}),"\n",(0,i.jsx)(n.h4,{id:"pause",children:"PAUSE"}),"\n",(0,i.jsx)(n.p,{children:"Pauses the task."}),"\n",(0,i.jsx)(n.h4,{id:"resume",children:"RESUME"}),"\n",(0,i.jsx)(n.p,{children:"Resumes the task."}),"\n",(0,i.jsx)(n.h4,{id:"stop",children:"STOP"}),"\n",(0,i.jsx)(n.p,{children:"Stops the task. Must send a start action to restart a new task."})]})}function h(e={}){const{wrapper:n}={...(0,s.R)(),...e.components};return n?(0,i.jsx)(n,{...e,children:(0,i.jsx)(d,{...e})}):d(e)}},8453:(e,n,t)=>{t.d(n,{R:()=>l,x:()=>o});var i=t(6540);const s={},a=i.createContext(s);function l(e){const n=i.useContext(a);return i.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function o(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(s):e.components||s:l(e.components),i.createElement(a.Provider,{value:n},e.children)}}}]);
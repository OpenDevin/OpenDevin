"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[2685],{5233:(e,n,l)=>{l.r(n),l.d(n,{assets:()=>d,contentTitle:()=>o,default:()=>a,frontMatter:()=>i,metadata:()=>t,toc:()=>c});var r=l(4848),s=l(8453);const i={sidebar_label:"logger",title:"logger"},o=void 0,t={id:"python/logger",title:"logger",description:"get\\console\\handler",source:"@site/docs/python/logger.md",sourceDirName:"python",slug:"/python/logger",permalink:"/docs/python/logger",draft:!1,unlisted:!1,tags:[],version:"current",frontMatter:{sidebar_label:"logger",title:"logger"}},d={},c=[{value:"get_console_handler",id:"get_console_handler",level:4},{value:"get_file_handler",id:"get_file_handler",level:4},{value:"log_uncaught_exceptions",id:"log_uncaught_exceptions",level:4},{value:"LlmFileHandler Objects",id:"llmfilehandler-objects",level:2},{value:"__init__",id:"__init__",level:4},{value:"emit",id:"emit",level:4},{value:"get_llm_prompt_file_handler",id:"get_llm_prompt_file_handler",level:4},{value:"get_llm_response_file_handler",id:"get_llm_response_file_handler",level:4}];function h(e){const n={code:"code",em:"em",h2:"h2",h4:"h4",li:"li",p:"p",pre:"pre",strong:"strong",ul:"ul",...(0,s.R)(),...e.components};return(0,r.jsxs)(r.Fragment,{children:[(0,r.jsx)(n.h4,{id:"get_console_handler",children:"get_console_handler"}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-python",children:"def get_console_handler()\n"})}),"\n",(0,r.jsx)(n.p,{children:"Returns a console handler for logging."}),"\n",(0,r.jsx)(n.h4,{id:"get_file_handler",children:"get_file_handler"}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-python",children:"def get_file_handler()\n"})}),"\n",(0,r.jsx)(n.p,{children:"Returns a file handler for logging."}),"\n",(0,r.jsx)(n.h4,{id:"log_uncaught_exceptions",children:"log_uncaught_exceptions"}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-python",children:"def log_uncaught_exceptions(ex_cls, ex, tb)\n"})}),"\n",(0,r.jsx)(n.p,{children:"Logs uncaught exceptions along with the traceback."}),"\n",(0,r.jsxs)(n.p,{children:[(0,r.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,r.jsxs)(n.ul,{children:["\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"ex_cls"})," ",(0,r.jsx)(n.em,{children:"type"})," - The type of the exception."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"ex"})," ",(0,r.jsx)(n.em,{children:"Exception"})," - The exception instance."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"tb"})," ",(0,r.jsx)(n.em,{children:"traceback"})," - The traceback object."]}),"\n"]}),"\n",(0,r.jsxs)(n.p,{children:[(0,r.jsx)(n.strong,{children:"Returns"}),":"]}),"\n",(0,r.jsx)(n.p,{children:"None"}),"\n",(0,r.jsx)(n.h2,{id:"llmfilehandler-objects",children:"LlmFileHandler Objects"}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-python",children:"class LlmFileHandler(logging.FileHandler)\n"})}),"\n",(0,r.jsx)(n.p,{children:(0,r.jsx)(n.strong,{children:"LLM prompt and response logging"})}),"\n",(0,r.jsx)(n.h4,{id:"__init__",children:"__init__"}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-python",children:"def __init__(filename, mode='a', encoding='utf-8', delay=False)\n"})}),"\n",(0,r.jsx)(n.p,{children:"Initializes an instance of LlmFileHandler."}),"\n",(0,r.jsxs)(n.p,{children:[(0,r.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,r.jsxs)(n.ul,{children:["\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"filename"})," ",(0,r.jsx)(n.em,{children:"str"})," - The name of the log file."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"mode"})," ",(0,r.jsx)(n.em,{children:"str, optional"})," - The file mode. Defaults to 'a'."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"encoding"})," ",(0,r.jsx)(n.em,{children:"str, optional"})," - The file encoding. Defaults to None."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"delay"})," ",(0,r.jsx)(n.em,{children:"bool, optional"})," - Whether to delay file opening. Defaults to False."]}),"\n"]}),"\n",(0,r.jsx)(n.h4,{id:"emit",children:"emit"}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-python",children:"def emit(record)\n"})}),"\n",(0,r.jsx)(n.p,{children:"Emits a log record."}),"\n",(0,r.jsxs)(n.p,{children:[(0,r.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,r.jsxs)(n.ul,{children:["\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"record"})," ",(0,r.jsx)(n.em,{children:"logging.LogRecord"})," - The log record to emit."]}),"\n"]}),"\n",(0,r.jsx)(n.h4,{id:"get_llm_prompt_file_handler",children:"get_llm_prompt_file_handler"}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-python",children:"def get_llm_prompt_file_handler()\n"})}),"\n",(0,r.jsx)(n.p,{children:"Returns a file handler for LLM prompt logging."}),"\n",(0,r.jsx)(n.h4,{id:"get_llm_response_file_handler",children:"get_llm_response_file_handler"}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-python",children:"def get_llm_response_file_handler()\n"})}),"\n",(0,r.jsx)(n.p,{children:"Returns a file handler for LLM response logging."})]})}function a(e={}){const{wrapper:n}={...(0,s.R)(),...e.components};return n?(0,r.jsx)(n,{...e,children:(0,r.jsx)(h,{...e})}):h(e)}},8453:(e,n,l)=>{l.d(n,{R:()=>o,x:()=>t});var r=l(6540);const s={},i=r.createContext(s);function o(e){const n=r.useContext(i);return r.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function t(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(s):e.components||s:o(e.components),r.createElement(i.Provider,{value:n},e.children)}}}]);
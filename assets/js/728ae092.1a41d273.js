"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[8267],{4951:(e,n,t)=>{t.r(n),t.d(n,{assets:()=>i,contentTitle:()=>l,default:()=>h,frontMatter:()=>r,metadata:()=>a,toc:()=>c});var o=t(4848),s=t(8453);const r={sidebar_label:"agent",title:"agenthub.monologue_agent.agent"},l=void 0,a={id:"python/agenthub/monologue_agent/agent",title:"agenthub.monologue_agent.agent",description:"MonologueAgent Objects",source:"@site/docs/python/agenthub/monologue_agent/agent.md",sourceDirName:"python/agenthub/monologue_agent",slug:"/python/agenthub/monologue_agent/agent",permalink:"/docs/python/agenthub/monologue_agent/agent",draft:!1,unlisted:!1,tags:[],version:"current",frontMatter:{sidebar_label:"agent",title:"agenthub.monologue_agent.agent"},sidebar:"apiSidebar",previous:{title:"prompts",permalink:"/docs/python/agenthub/monologue_agent/utils/prompts"},next:{title:"agent",permalink:"/docs/python/agenthub/planner_agent/agent"}},i={},c=[{value:"MonologueAgent Objects",id:"monologueagent-objects",level:2},{value:"__init__",id:"__init__",level:4},{value:"step",id:"step",level:4},{value:"search_memory",id:"search_memory",level:4}];function d(e){const n={code:"code",h2:"h2",h4:"h4",li:"li",p:"p",pre:"pre",strong:"strong",ul:"ul",...(0,s.R)(),...e.components};return(0,o.jsxs)(o.Fragment,{children:[(0,o.jsx)(n.h2,{id:"monologueagent-objects",children:"MonologueAgent Objects"}),"\n",(0,o.jsx)(n.pre,{children:(0,o.jsx)(n.code,{className:"language-python",children:"class MonologueAgent(Agent)\n"})}),"\n",(0,o.jsx)(n.p,{children:"The Monologue Agent utilizes long and short term memory to complete tasks.\nLong term memory is stored as a LongTermMemory object and the model uses it to search for examples from the past.\nShort term memory is stored as a Monologue object and the model can condense it as necessary."}),"\n",(0,o.jsx)(n.h4,{id:"__init__",children:"__init__"}),"\n",(0,o.jsx)(n.pre,{children:(0,o.jsx)(n.code,{className:"language-python",children:"def __init__(llm: LLM)\n"})}),"\n",(0,o.jsx)(n.p,{children:"Initializes the Monologue Agent with an llm, monologue, and memory."}),"\n",(0,o.jsxs)(n.p,{children:[(0,o.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,o.jsxs)(n.ul,{children:["\n",(0,o.jsx)(n.li,{children:"llm (LLM): The llm to be used by this agent"}),"\n"]}),"\n",(0,o.jsx)(n.h4,{id:"step",children:"step"}),"\n",(0,o.jsx)(n.pre,{children:(0,o.jsx)(n.code,{className:"language-python",children:"def step(state: State) -> Action\n"})}),"\n",(0,o.jsx)(n.p,{children:"Modifies the current state by adding the most recent actions and observations, then prompts the model to think about it's next action to take using monologue, memory, and hint."}),"\n",(0,o.jsxs)(n.p,{children:[(0,o.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,o.jsxs)(n.ul,{children:["\n",(0,o.jsx)(n.li,{children:"state (State): The current state based on previous steps taken"}),"\n"]}),"\n",(0,o.jsxs)(n.p,{children:[(0,o.jsx)(n.strong,{children:"Returns"}),":"]}),"\n",(0,o.jsxs)(n.ul,{children:["\n",(0,o.jsx)(n.li,{children:"Action: The next action to take based on LLM response"}),"\n"]}),"\n",(0,o.jsx)(n.h4,{id:"search_memory",children:"search_memory"}),"\n",(0,o.jsx)(n.pre,{children:(0,o.jsx)(n.code,{className:"language-python",children:"def search_memory(query: str) -> List[str]\n"})}),"\n",(0,o.jsx)(n.p,{children:"Uses VectorIndexRetriever to find related memories within the long term memory.\nUses search to produce top 10 results."}),"\n",(0,o.jsxs)(n.p,{children:[(0,o.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,o.jsxs)(n.ul,{children:["\n",(0,o.jsx)(n.li,{children:"query (str): The query that we want to find related memories for"}),"\n"]}),"\n",(0,o.jsxs)(n.p,{children:[(0,o.jsx)(n.strong,{children:"Returns"}),":"]}),"\n",(0,o.jsxs)(n.ul,{children:["\n",(0,o.jsx)(n.li,{children:"List[str]: A list of top 10 text results that matched the query"}),"\n"]})]})}function h(e={}){const{wrapper:n}={...(0,s.R)(),...e.components};return n?(0,o.jsx)(n,{...e,children:(0,o.jsx)(d,{...e})}):d(e)}},8453:(e,n,t)=>{t.d(n,{R:()=>l,x:()=>a});var o=t(6540);const s={},r=o.createContext(s);function l(e){const n=o.useContext(r);return o.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function a(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(s):e.components||s:l(e.components),o.createElement(r.Provider,{value:n},e.children)}}}]);
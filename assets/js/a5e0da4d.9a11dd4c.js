"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[5527],{3409:(e,n,o)=>{o.r(n),o.d(n,{assets:()=>c,contentTitle:()=>i,default:()=>h,frontMatter:()=>t,metadata:()=>l,toc:()=>d});var s=o(4848),r=o(8453);const t={sidebar_position:4},i="\ud83d\udea7 Troubleshooting",l={id:"usage/troubleshooting/troubleshooting",title:"\ud83d\udea7 Troubleshooting",description:"There are some error messages that get reported over and over by users.",source:"@site/docs/usage/troubleshooting/troubleshooting.md",sourceDirName:"usage/troubleshooting",slug:"/usage/troubleshooting/",permalink:"/docs/usage/troubleshooting/",draft:!1,unlisted:!1,tags:[],version:"current",sidebarPosition:4,frontMatter:{sidebar_position:4},sidebar:"docsSidebar",previous:{title:"\ud83e\udde0 Agents and Capabilities",permalink:"/docs/usage/agents"},next:{title:"Notes for Windows and WSL Users",permalink:"/docs/usage/troubleshooting/windows"}},c={},d=[{value:"Unable to connect to docker",id:"unable-to-connect-to-docker",level:2},{value:"Symptoms",id:"symptoms",level:3},{value:"Details",id:"details",level:3},{value:"Workarounds",id:"workarounds",level:3},{value:"Unable to connect to SSH box",id:"unable-to-connect-to-ssh-box",level:2},{value:"Symptoms",id:"symptoms-1",level:3},{value:"Details",id:"details-1",level:3},{value:"Workarounds",id:"workarounds-1",level:3},{value:"Unable to connect to LLM",id:"unable-to-connect-to-llm",level:2},{value:"Symptoms",id:"symptoms-2",level:3},{value:"Details",id:"details-2",level:3},{value:"Workarounds",id:"workarounds-2",level:3}];function a(e){const n={a:"a",admonition:"admonition",code:"code",h1:"h1",h2:"h2",h3:"h3",li:"li",p:"p",pre:"pre",strong:"strong",ul:"ul",...(0,r.R)(),...e.components};return(0,s.jsxs)(s.Fragment,{children:[(0,s.jsx)(n.h1,{id:"-troubleshooting",children:"\ud83d\udea7 Troubleshooting"}),"\n",(0,s.jsx)(n.p,{children:"There are some error messages that get reported over and over by users.\nWe'll try and make the install process easier, and to make these error messages\nbetter in the future. But for now, you can look for your error message below,\nand see if there are any workaround."}),"\n",(0,s.jsxs)(n.p,{children:["For each of these error messages ",(0,s.jsx)(n.strong,{children:"there is an existing issue"}),". Please do not\nopen an new issue--just comment there."]}),"\n",(0,s.jsx)(n.p,{children:"If you find more information or a workaround for one of these issues, please\nopen a PR to add details to this file."}),"\n",(0,s.jsx)(n.admonition,{type:"tip",children:(0,s.jsxs)(n.p,{children:["If you're running on Windows and having trouble, check out our ",(0,s.jsx)(n.a,{href:"troubleshooting/windows",children:"guide for Windows users"})]})}),"\n",(0,s.jsx)(n.h2,{id:"unable-to-connect-to-docker",children:(0,s.jsx)(n.a,{href:"https://github.com/OpenDevin/OpenDevin/issues/1226",children:"Unable to connect to docker"})}),"\n",(0,s.jsx)(n.h3,{id:"symptoms",children:"Symptoms"}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{children:"Error creating controller. Please check Docker is running using docker ps\n"})}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{children:"docker.errors.DockerException: Error while fetching server API version: ('Connection aborted.', FileNotFoundError(2, 'No such file or directory'))\n"})}),"\n",(0,s.jsx)(n.h3,{id:"details",children:"Details"}),"\n",(0,s.jsx)(n.p,{children:"OpenDevin uses a docker container to do its work safely, without potentially breaking your machine."}),"\n",(0,s.jsx)(n.h3,{id:"workarounds",children:"Workarounds"}),"\n",(0,s.jsxs)(n.ul,{children:["\n",(0,s.jsxs)(n.li,{children:["Run ",(0,s.jsx)(n.code,{children:"docker ps"})," to ensure that docker is running"]}),"\n",(0,s.jsxs)(n.li,{children:["Make sure you don't need ",(0,s.jsx)(n.code,{children:"sudo"})," to run docker ",(0,s.jsx)(n.a,{href:"https://www.baeldung.com/linux/docker-run-without-sudo",children:"see here"})]}),"\n"]}),"\n",(0,s.jsx)(n.h2,{id:"unable-to-connect-to-ssh-box",children:(0,s.jsx)(n.a,{href:"https://github.com/OpenDevin/OpenDevin/issues/1156",children:"Unable to connect to SSH box"})}),"\n",(0,s.jsx)(n.h3,{id:"symptoms-1",children:"Symptoms"}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{children:"self.shell = DockerSSHBox(\n...\npexpect.pxssh.ExceptionPxssh: Could not establish connection to host\n"})}),"\n",(0,s.jsx)(n.h3,{id:"details-1",children:"Details"}),"\n",(0,s.jsx)(n.p,{children:"By default, OpenDevin connects to a running container using SSH. On some machines,\nespecially Windows, this seems to fail."}),"\n",(0,s.jsx)(n.h3,{id:"workarounds-1",children:"Workarounds"}),"\n",(0,s.jsxs)(n.ul,{children:["\n",(0,s.jsx)(n.li,{children:"Restart your computer (sometimes works?)"}),"\n",(0,s.jsx)(n.li,{children:"Be sure to have the latest versions of WSL and Docker"}),"\n",(0,s.jsxs)(n.li,{children:["Try ",(0,s.jsx)(n.a,{href:"https://github.com/OpenDevin/OpenDevin/issues/1156#issuecomment-2064549427",children:"this reinstallation guide"})]}),"\n",(0,s.jsxs)(n.li,{children:["Set ",(0,s.jsx)(n.code,{children:"-e SANDBOX_TYPE=exec"})," to switch to the ExecBox docker container"]}),"\n"]}),"\n",(0,s.jsx)(n.h2,{id:"unable-to-connect-to-llm",children:(0,s.jsx)(n.a,{href:"https://github.com/OpenDevin/OpenDevin/issues/1208",children:"Unable to connect to LLM"})}),"\n",(0,s.jsx)(n.h3,{id:"symptoms-2",children:"Symptoms"}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{children:"  File \"/app/.venv/lib/python3.12/site-packages/openai/_exceptions.py\", line 81, in __init__\n    super().__init__(message, response.request, body=body)\n                              ^^^^^^^^^^^^^^^^\nAttributeError: 'NoneType' object has no attribute 'request'\n"})}),"\n",(0,s.jsx)(n.h3,{id:"details-2",children:"Details"}),"\n",(0,s.jsxs)(n.p,{children:["This usually happens with local LLM setups, when OpenDevin can't connect to the LLM server.\nSee our guide for ",(0,s.jsx)(n.a,{href:"llms/localLLMs",children:"local LLMs"})," for more information."]}),"\n",(0,s.jsx)(n.h3,{id:"workarounds-2",children:"Workarounds"}),"\n",(0,s.jsxs)(n.ul,{children:["\n",(0,s.jsxs)(n.li,{children:["Check your ",(0,s.jsx)(n.code,{children:"LLM_BASE_URL"})]}),"\n",(0,s.jsx)(n.li,{children:"Check that ollama is running OK"}),"\n",(0,s.jsxs)(n.li,{children:["Make sure you're using ",(0,s.jsx)(n.code,{children:"--add-host host.docker.internal=host-gateway"})," when running in docker"]}),"\n"]})]})}function h(e={}){const{wrapper:n}={...(0,r.R)(),...e.components};return n?(0,s.jsx)(n,{...e,children:(0,s.jsx)(a,{...e})}):a(e)}},8453:(e,n,o)=>{o.d(n,{R:()=>i,x:()=>l});var s=o(6540);const r={},t=s.createContext(r);function i(e){const n=s.useContext(t);return s.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function l(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(r):e.components||r:i(e.components),s.createElement(t.Provider,{value:n},e.children)}}}]);
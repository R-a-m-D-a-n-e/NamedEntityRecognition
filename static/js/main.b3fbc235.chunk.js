(this.webpackJsonpner=this.webpackJsonpner||[]).push([[0],{104:function(t,e,n){},105:function(t,e,n){},106:function(t,e,n){},107:function(t,e,n){},108:function(t,e,n){},109:function(t,e,n){},110:function(t,e,n){"use strict";n.r(e);var a=n(0),c=n.n(a),i=n(28),r=n.n(i),s=(n(79),n(13)),o=n(14),l=n(21),u=n(20),h=n(29),j=n(12),b=n(147),d=n(150),O=n(54),p=n(149),f=n(151),g=n(148),x=n(2),v=Object(b.a)((function(t){return{root:{display:"flex",justifyContent:"center",alignItems:"center",width:100,height:"2.5em",padding:"5px",border:"1px solid #000",boxShadow:"0px 2px 14px #000",borderRadius:"5px",color:"#fff1e6",overflow:"hidden"}}})),y=["Allemand","Anglais","Espagnol","Fran\xe7ais","Nerlande"];function m(t){var e=v(),n=c.a.useState(null),a=Object(j.a)(n,2),i=a[0],r=a[1],s=c.a.useState(2),o=Object(j.a)(s,2),l=o[0],u=o[1];return Object(x.jsxs)("div",{className:e.root,children:[Object(x.jsx)(d.a,{component:"nav","aria-label":"Device settings",children:Object(x.jsx)(O.b,{button:!0,"aria-haspopup":"true","aria-controls":"lock-menu","aria-label":"Langue",onClick:function(t){r(t.currentTarget)},children:Object(x.jsx)(p.a,{primary:"Langue",secondary:y[l]})})}),Object(x.jsx)(g.a,{id:"lock-menu",anchorEl:i,keepMounted:!0,open:Boolean(i),onClose:function(){r(null)},children:y.map((function(e,n){return Object(x.jsx)(f.a,{selected:n===l,onClick:function(e){!function(e,n){t.changeLangue(y[n]),u(n),r(null)}(0,n)},children:e},e)}))})]})}var k=function(t){Object(l.a)(n,t);var e=Object(u.a)(n);function n(){return Object(s.a)(this,n),e.apply(this,arguments)}return Object(o.a)(n,[{key:"render",value:function(){return Object(x.jsx)("div",{className:"button",children:Object(x.jsx)("input",{id:"button",type:"submit",value:"Extract"})})}}]),n}(a.Component),C=function(t){Object(l.a)(n,t);var e=Object(u.a)(n);function n(t){return Object(s.a)(this,n),e.call(this,t)}return Object(o.a)(n,[{key:"render",value:function(){return Object(x.jsx)("div",{id:"wrapper",children:Object(x.jsx)("textarea",{onChange:this.props.onChange,value:this.props.value,placeholder:"Enter something funny.",id:"text",name:"text",rows:"4",style:{overflow:"hidden",wordWrap:"break-word",resize:"none",height:"160px"}})})}}]),n}(a.Component),E=n(68),L=n.n(E),S=function(){function t(){Object(s.a)(this,t)}return Object(o.a)(t,null,[{key:"extract",value:function(t,e,n,a){var c=JSON.stringify({opt:t,text:e});L.a.post("http://127.0.0.1:5000/extract",c,{headers:{"Content-Type":"application/json"}}).then((function(t){var e=t.data.entitys;"ok"===t.data.state?n(e):a([])})).catch((function(){a([])}))}}]),t}(),N=function(t){Object(l.a)(n,t);var e=Object(u.a)(n);function n(t){var a;return Object(s.a)(this,n),(a=e.call(this,t)).state={value:"",lng:"Espagnol"},a.handleChange=a.handleChange.bind(Object(h.a)(a)),a.handleSubmit=a.handleSubmit.bind(Object(h.a)(a)),a.changeLangue=a.changeLangue.bind(Object(h.a)(a)),a}return Object(o.a)(n,[{key:"changeLangue",value:function(t){this.setState({lng:t})}},{key:"handleChange",value:function(t){this.setState({value:t.target.value})}},{key:"handleSubmit",value:function(t){t.preventDefault(),S.extract(this.state.lng,this.state.value,this.props.changeListEntity,this.props.changeListEntity),console.log("ooooook")}},{key:"render",value:function(){return Object(x.jsx)("div",{children:Object(x.jsxs)("form",{className:"form",method:"get",action:"",onSubmit:this.handleSubmit,children:[Object(x.jsx)(m,{changeLangue:this.changeLangue}),Object(x.jsx)(C,{onChange:this.handleChange,value:this.state.value}),Object(x.jsx)(k,{})]})})}}]),n}(a.Component),w=function(t){Object(l.a)(n,t);var e=Object(u.a)(n);function n(){return Object(s.a)(this,n),e.apply(this,arguments)}return Object(o.a)(n,[{key:"render",value:function(){var t=[];return this.props.listEntity.forEach((function(e,n){t.push(Object(x.jsxs)("tr",{children:[Object(x.jsx)("td",{children:e.tag}),Object(x.jsx)("td",{children:e.value})]},n))})),console.log(t),Object(x.jsx)("div",{className:"namedentity",children:Object(x.jsx)("div",{className:"namedentity-cont",children:Object(x.jsxs)("table",{className:"styled-table",children:[Object(x.jsx)("thead",{children:Object(x.jsxs)("tr",{children:[Object(x.jsx)("th",{children:"Tag"}),Object(x.jsx)("th",{children:"Value"})]})}),Object(x.jsx)("tbody",{children:t})]})})})}}]),n}(a.Component),F=function(t){Object(l.a)(n,t);var e=Object(u.a)(n);function n(t){var a;return Object(s.a)(this,n),(a=e.call(this,t)).state={listEntity:[]},a.changeListEntity=a.changeListEntity.bind(Object(h.a)(a)),a}return Object(o.a)(n,[{key:"changeListEntity",value:function(t){this.setState({listEntity:t})}},{key:"render",value:function(){return Object(x.jsx)("div",{className:"ner",children:Object(x.jsxs)("div",{className:"ner-cont",children:[Object(x.jsx)(N,{changeListEntity:this.changeListEntity}),Object(x.jsx)(w,{listEntity:this.state.listEntity})]})})}}]),n}(a.Component),T=(n(104),n(105),n(106),n(107),n(108),n(109),function(t){Object(l.a)(n,t);var e=Object(u.a)(n);function n(){return Object(s.a)(this,n),e.apply(this,arguments)}return Object(o.a)(n,[{key:"render",value:function(){return Object(x.jsx)("div",{className:"App",children:Object(x.jsx)(F,{})})}}]),n}(a.Component)),A=function(t){t&&t instanceof Function&&n.e(3).then(n.bind(null,152)).then((function(e){var n=e.getCLS,a=e.getFID,c=e.getFCP,i=e.getLCP,r=e.getTTFB;n(t),a(t),c(t),i(t),r(t)}))};r.a.render(Object(x.jsx)(c.a.StrictMode,{children:Object(x.jsx)(T,{})}),document.getElementById("root")),A()},79:function(t,e,n){}},[[110,1,2]]]);
//# sourceMappingURL=main.b3fbc235.chunk.js.map
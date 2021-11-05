import axios from "axios";

export default class Extract {
  static extract(opt, text, callback, callbackcatch) {
    // console.log(opt, text);
    const json = JSON.stringify({
      opt,
      text,
    });
    axios
      .post(`http://127.0.0.1:5000/extract`, json, {
        headers: {
          "Content-Type": "application/json",
        },
      })
      .then((res) => {
        // console.log(res);
        let entitys = res.data.entitys;
        // console.log(entitys);
        if (res.data.state === "ok") callback(entitys);
        else callbackcatch([]);
      })
      .catch(() => {
        callbackcatch([]);
      });
  }
}

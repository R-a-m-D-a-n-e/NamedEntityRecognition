import { Component } from "react";
import Ner from "./Component/Ner";
import "./Style/App.css";
import "./Style/text.css";
import "./Style/namedentity.css";
import "./Style/ner.css";
import "./Style/button.css";
import "./Style/form.css";

export default class App extends Component {
  render() {
    return (
      <div className="App">
        <Ner />
      </div>
    );
  }
}

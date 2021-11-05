import React, { Component } from "react";
import Form from "./Form";
import NamedEntity from "./NamedEntity";
import ReactLoading from "react-loading";

export default class Ner extends Component {
  constructor(props) {
    super(props);
    this.state = {
      listEntity: [],
      load: false,
    };
    this.changeListEntity = this.changeListEntity.bind(this);
    this.setLoad = this.setLoad.bind(this);
  }

  changeListEntity(newlistEntity) {
    this.setState({ listEntity: newlistEntity, load: false });
  }

  setLoad(val) {
    this.setState({ load: val });
  }

  render() {
    return (
      <div className="ner">
        <span className="ner-title">Named Entity Recognition</span>
        <div className="ner-cont">
          {this.state.load ? (
            <div className="ner-load">
              <ReactLoading
                type={"spin"}
                color={"#adb5bd"}
                height={"5em"}
                width={"5em"}
                className="load"
              />
            </div>
          ) : (
            ""
          )}
          <Form
            changeListEntity={this.changeListEntity}
            setLoad={this.setLoad}
          />
          <NamedEntity listEntity={this.state.listEntity} />
        </div>
      </div>
    );
  }
}

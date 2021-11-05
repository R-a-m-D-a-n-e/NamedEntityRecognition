import React, { Component } from "react";
import MenuLangue from "./MenuLangue";
import Button from "./Button";
import Text from "./Text";
import Extract from "../Object/Extract";

export default class Form extends Component {
  constructor(props) {
    super(props);
    this.state = {
      value: "",
      lng: "Espagnol",
    };
    this.handleChange = this.handleChange.bind(this);
    this.handleSubmit = this.handleSubmit.bind(this);
    this.changeLangue = this.changeLangue.bind(this);
  }
  changeLangue(newlng) {
    this.setState({ lng: newlng });
  }
  handleChange(e) {
    this.setState({ value: e.target.value });
  }
  handleSubmit(e) {
    e.preventDefault();
    this.props.setLoad(true);
    setTimeout(() => {
      Extract.extract(
        this.state.lng,
        this.state.value,
        this.props.changeListEntity,
        this.props.changeListEntity
      );
    }, 5);
  }
  render() {
    return (
      <div>
        <form
          className="form"
          method="get"
          action=""
          onSubmit={this.handleSubmit}
        >
          <MenuLangue changeLangue={this.changeLangue} />
          <Text onChange={this.handleChange} value={this.state.value} />
          <Button />
        </form>
      </div>
    );
  }
}

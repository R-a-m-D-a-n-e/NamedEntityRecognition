import React, { Component } from "react";

export default class Button extends Component {
  render() {
    return (
      <div className="button">
        <input id="button" type="submit" value="Extract" />
      </div>
    );
  }
}

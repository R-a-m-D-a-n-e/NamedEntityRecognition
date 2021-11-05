import React, { Component } from "react";

export default class Text extends Component {
  render() {
    return (
      <div id="wrapper">
        <textarea
          onChange={this.props.onChange}
          value={this.props.value}
          placeholder="Enter something funny."
          id="text"
          name="text"
          rows="4"
          style={{
            overflow: "hidden",
            wordWrap: "break-word",
            resize: "none",
            height: "160px",
          }}
        />
      </div>
    );
  }
}

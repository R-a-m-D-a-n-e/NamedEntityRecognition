import React, { Component } from "react";

export default class NamedEntity extends Component {
  render() {
    let trs = [];
    this.props.listEntity.forEach((e, i) => {
      trs.push(
        <tr key={i}>
          <td>{e.tag}</td>
          <td>{e.value}</td>
        </tr>
      );
    });
    // console.log(trs);
    return (
      <div className="namedentity">
        <div className="namedentity-cont">
          <table className="styled-table">
            <thead>
              <tr>
                <th>Tag</th>
                <th>Value</th>
              </tr>
            </thead>
            <tbody>{trs}</tbody>
          </table>
        </div>
      </div>
    );
  }
}

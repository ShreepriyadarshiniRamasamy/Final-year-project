function _typeof(obj) { if (typeof Symbol === "function" && typeof Symbol.iterator === "symbol") { _typeof = function _typeof(obj) { return typeof obj; }; } else { _typeof = function _typeof(obj) { return obj && typeof Symbol === "function" && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj; }; } return _typeof(obj); }

function ownKeys(object, enumerableOnly) { var keys = Object.keys(object); if (Object.getOwnPropertySymbols) { var symbols = Object.getOwnPropertySymbols(object); if (enumerableOnly) symbols = symbols.filter(function (sym) { return Object.getOwnPropertyDescriptor(object, sym).enumerable; }); keys.push.apply(keys, symbols); } return keys; }

function _objectSpread(target) { for (var i = 1; i < arguments.length; i++) { var source = arguments[i] != null ? arguments[i] : {}; if (i % 2) { ownKeys(source, true).forEach(function (key) { _defineProperty(target, key, source[key]); }); } else if (Object.getOwnPropertyDescriptors) { Object.defineProperties(target, Object.getOwnPropertyDescriptors(source)); } else { ownKeys(source).forEach(function (key) { Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key)); }); } } return target; }

function _defineProperty(obj, key, value) { if (key in obj) { Object.defineProperty(obj, key, { value: value, enumerable: true, configurable: true, writable: true }); } else { obj[key] = value; } return obj; }

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

function _defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } }

function _createClass(Constructor, protoProps, staticProps) { if (protoProps) _defineProperties(Constructor.prototype, protoProps); if (staticProps) _defineProperties(Constructor, staticProps); return Constructor; }

function _possibleConstructorReturn(self, call) { if (call && (_typeof(call) === "object" || typeof call === "function")) { return call; } return _assertThisInitialized(self); }

function _getPrototypeOf(o) { _getPrototypeOf = Object.setPrototypeOf ? Object.getPrototypeOf : function _getPrototypeOf(o) { return o.__proto__ || Object.getPrototypeOf(o); }; return _getPrototypeOf(o); }

function _assertThisInitialized(self) { if (self === void 0) { throw new ReferenceError("this hasn't been initialised - super() hasn't been called"); } return self; }

function _inherits(subClass, superClass) { if (typeof superClass !== "function" && superClass !== null) { throw new TypeError("Super expression must either be null or a function"); } subClass.prototype = Object.create(superClass && superClass.prototype, { constructor: { value: subClass, writable: true, configurable: true } }); if (superClass) _setPrototypeOf(subClass, superClass); }

function _setPrototypeOf(o, p) { _setPrototypeOf = Object.setPrototypeOf || function _setPrototypeOf(o, p) { o.__proto__ = p; return o; }; return _setPrototypeOf(o, p); }

import React from 'react';
import ReactDOM from 'react-dom';
import KeyCode from "rc-util/es/KeyCode";
import classNames from 'classnames';
import scrollIntoView from 'dom-scroll-into-view';
import { connect } from 'mini-store';
import { noop, menuAllProps } from './util';
export var MenuItem =
/*#__PURE__*/
function (_React$Component) {
  _inherits(MenuItem, _React$Component);

  function MenuItem() {
    var _this;

    _classCallCheck(this, MenuItem);

    _this = _possibleConstructorReturn(this, _getPrototypeOf(MenuItem).apply(this, arguments));

    _this.onKeyDown = function (e) {
      var keyCode = e.keyCode;

      if (keyCode === KeyCode.ENTER) {
        _this.onClick(e);

        return true;
      }

      return undefined;
    };

    _this.onMouseLeave = function (e) {
      var _this$props = _this.props,
          eventKey = _this$props.eventKey,
          onItemHover = _this$props.onItemHover,
          onMouseLeave = _this$props.onMouseLeave;
      onItemHover({
        key: eventKey,
        hover: false
      });
      onMouseLeave({
        key: eventKey,
        domEvent: e
      });
    };

    _this.onMouseEnter = function (e) {
      var _this$props2 = _this.props,
          eventKey = _this$props2.eventKey,
          onItemHover = _this$props2.onItemHover,
          onMouseEnter = _this$props2.onMouseEnter;
      onItemHover({
        key: eventKey,
        hover: true
      });
      onMouseEnter({
        key: eventKey,
        domEvent: e
      });
    };

    _this.onClick = function (e) {
      var _this$props3 = _this.props,
          eventKey = _this$props3.eventKey,
          multiple = _this$props3.multiple,
          onClick = _this$props3.onClick,
          onSelect = _this$props3.onSelect,
          onDeselect = _this$props3.onDeselect,
          isSelected = _this$props3.isSelected;
      var info = {
        key: eventKey,
        keyPath: [eventKey],
        item: _assertThisInitialized(_this),
        domEvent: e
      };
      onClick(info);

      if (multiple) {
        if (isSelected) {
          onDeselect(info);
        } else {
          onSelect(info);
        }
      } else if (!isSelected) {
        onSelect(info);
      }
    };

    _this.saveNode = function (node) {
      _this.node = node;
    };

    return _this;
  }

  _createClass(MenuItem, [{
    key: "componentDidMount",
    value: function componentDidMount() {
      // invoke customized ref to expose component to mixin
      this.callRef();
    }
  }, {
    key: "componentDidUpdate",
    value: function componentDidUpdate(prevProps) {
      var _this$props4 = this.props,
          active = _this$props4.active,
          parentMenu = _this$props4.parentMenu,
          eventKey = _this$props4.eventKey; // 在 parentMenu 上层保存滚动状态，避免重复的 MenuItem key 导致滚动跳动
      // https://github.com/ant-design/ant-design/issues/16181

      if (!prevProps.active && active && (!parentMenu || !parentMenu["scrolled-".concat(eventKey)])) {
        if (this.node) {
          scrollIntoView(this.node, ReactDOM.findDOMNode(parentMenu), {
            onlyScrollIfNeeded: true
          });
          parentMenu["scrolled-".concat(eventKey)] = true;
        }
      } else if (parentMenu && parentMenu["scrolled-".concat(eventKey)]) {
        delete parentMenu["scrolled-".concat(eventKey)];
      }

      this.callRef();
    }
  }, {
    key: "componentWillUnmount",
    value: function componentWillUnmount() {
      var props = this.props;

      if (props.onDestroy) {
        props.onDestroy(props.eventKey);
      }
    }
  }, {
    key: "getPrefixCls",
    value: function getPrefixCls() {
      return "".concat(this.props.rootPrefixCls, "-item");
    }
  }, {
    key: "getActiveClassName",
    value: function getActiveClassName() {
      return "".concat(this.getPrefixCls(), "-active");
    }
  }, {
    key: "getSelectedClassName",
    value: function getSelectedClassName() {
      return "".concat(this.getPrefixCls(), "-selected");
    }
  }, {
    key: "getDisabledClassName",
    value: function getDisabledClassName() {
      return "".concat(this.getPrefixCls(), "-disabled");
    }
  }, {
    key: "callRef",
    value: function callRef() {
      if (this.props.manualRef) {
        this.props.manualRef(this);
      }
    }
  }, {
    key: "render",
    value: function render() {
      var _classNames;

      var props = _objectSpread({}, this.props);

      var className = classNames(this.getPrefixCls(), props.className, (_classNames = {}, _defineProperty(_classNames, this.getActiveClassName(), !props.disabled && props.active), _defineProperty(_classNames, this.getSelectedClassName(), props.isSelected), _defineProperty(_classNames, this.getDisabledClassName(), props.disabled), _classNames));

      var attrs = _objectSpread({}, props.attribute, {
        title: props.title,
        className: className,
        // set to menuitem by default
        role: props.role || 'menuitem',
        'aria-disabled': props.disabled
      });

      if (props.role === 'option') {
        // overwrite to option
        attrs = _objectSpread({}, attrs, {
          role: 'option',
          'aria-selected': props.isSelected
        });
      } else if (props.role === null || props.role === 'none') {
        // sometimes we want to specify role inside <li/> element
        // <li><a role='menuitem'>Link</a></li> would be a good example
        // in this case the role on <li/> should be "none" to
        // remove the implied listitem role.
        // https://www.w3.org/TR/wai-aria-practices-1.1/examples/menubar/menubar-1/menubar-1.html
        attrs.role = 'none';
      } // In case that onClick/onMouseLeave/onMouseEnter is passed down from owner


      var mouseEvent = {
        onClick: props.disabled ? null : this.onClick,
        onMouseLeave: props.disabled ? null : this.onMouseLeave,
        onMouseEnter: props.disabled ? null : this.onMouseEnter
      };

      var style = _objectSpread({}, props.style);

      if (props.mode === 'inline') {
        style.paddingLeft = props.inlineIndent * props.level;
      }

      menuAllProps.forEach(function (key) {
        return delete props[key];
      });
      var icon = this.props.itemIcon;

      if (typeof this.props.itemIcon === 'function') {
        // TODO: This is a bug which should fixed after TS refactor
        icon = React.createElement(this.props.itemIcon, this.props);
      }

      return React.createElement("li", Object.assign({}, props, attrs, mouseEvent, {
        style: style,
        ref: this.saveNode
      }), props.children, icon);
    }
  }]);

  return MenuItem;
}(React.Component);
MenuItem.isMenuItem = true;
MenuItem.defaultProps = {
  onSelect: noop,
  onMouseEnter: noop,
  onMouseLeave: noop,
  manualRef: noop
};
var connected = connect(function (_ref, _ref2) {
  var activeKey = _ref.activeKey,
      selectedKeys = _ref.selectedKeys;
  var eventKey = _ref2.eventKey,
      subMenuKey = _ref2.subMenuKey;
  return {
    active: activeKey[subMenuKey] === eventKey,
    isSelected: selectedKeys.indexOf(eventKey) !== -1
  };
})(MenuItem);
export default connected;
"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports["default"] = void 0;

var React = _interopRequireWildcard(require("react"));

var PropTypes = _interopRequireWildcard(require("prop-types"));

var _rcDrawer = _interopRequireDefault(require("rc-drawer"));

var _createReactContext = _interopRequireDefault(require("@ant-design/create-react-context"));

var _warning = _interopRequireDefault(require("../_util/warning"));

var _classnames = _interopRequireDefault(require("classnames"));

var _icon = _interopRequireDefault(require("../icon"));

var _configProvider = require("../config-provider");

var _type = require("../_util/type");

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { "default": obj }; }

function _interopRequireWildcard(obj) { if (obj && obj.__esModule) { return obj; } else { var newObj = {}; if (obj != null) { for (var key in obj) { if (Object.prototype.hasOwnProperty.call(obj, key)) { var desc = Object.defineProperty && Object.getOwnPropertyDescriptor ? Object.getOwnPropertyDescriptor(obj, key) : {}; if (desc.get || desc.set) { Object.defineProperty(newObj, key, desc); } else { newObj[key] = obj[key]; } } } } newObj["default"] = obj; return newObj; } }

function _typeof(obj) { if (typeof Symbol === "function" && typeof Symbol.iterator === "symbol") { _typeof = function _typeof(obj) { return typeof obj; }; } else { _typeof = function _typeof(obj) { return obj && typeof Symbol === "function" && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj; }; } return _typeof(obj); }

function _extends() { _extends = Object.assign || function (target) { for (var i = 1; i < arguments.length; i++) { var source = arguments[i]; for (var key in source) { if (Object.prototype.hasOwnProperty.call(source, key)) { target[key] = source[key]; } } } return target; }; return _extends.apply(this, arguments); }

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

function _defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } }

function _createClass(Constructor, protoProps, staticProps) { if (protoProps) _defineProperties(Constructor.prototype, protoProps); if (staticProps) _defineProperties(Constructor, staticProps); return Constructor; }

function _possibleConstructorReturn(self, call) { if (call && (_typeof(call) === "object" || typeof call === "function")) { return call; } return _assertThisInitialized(self); }

function _getPrototypeOf(o) { _getPrototypeOf = Object.setPrototypeOf ? Object.getPrototypeOf : function _getPrototypeOf(o) { return o.__proto__ || Object.getPrototypeOf(o); }; return _getPrototypeOf(o); }

function _assertThisInitialized(self) { if (self === void 0) { throw new ReferenceError("this hasn't been initialised - super() hasn't been called"); } return self; }

function _inherits(subClass, superClass) { if (typeof superClass !== "function" && superClass !== null) { throw new TypeError("Super expression must either be null or a function"); } subClass.prototype = Object.create(superClass && superClass.prototype, { constructor: { value: subClass, writable: true, configurable: true } }); if (superClass) _setPrototypeOf(subClass, superClass); }

function _setPrototypeOf(o, p) { _setPrototypeOf = Object.setPrototypeOf || function _setPrototypeOf(o, p) { o.__proto__ = p; return o; }; return _setPrototypeOf(o, p); }

var __rest = void 0 && (void 0).__rest || function (s, e) {
  var t = {};

  for (var p in s) {
    if (Object.prototype.hasOwnProperty.call(s, p) && e.indexOf(p) < 0) t[p] = s[p];
  }

  if (s != null && typeof Object.getOwnPropertySymbols === "function") for (var i = 0, p = Object.getOwnPropertySymbols(s); i < p.length; i++) {
    if (e.indexOf(p[i]) < 0) t[p[i]] = s[p[i]];
  }
  return t;
};

var DrawerContext = (0, _createReactContext["default"])(null);
var PlacementTypes = (0, _type.tuple)('top', 'right', 'bottom', 'left');

var Drawer =
/*#__PURE__*/
function (_React$Component) {
  _inherits(Drawer, _React$Component);

  function Drawer() {
    var _this;

    _classCallCheck(this, Drawer);

    _this = _possibleConstructorReturn(this, _getPrototypeOf(Drawer).apply(this, arguments));
    _this.state = {
      push: false
    };

    _this.close = function (e) {
      if (_this.props.visible !== undefined) {
        if (_this.props.onClose) {
          _this.props.onClose(e);
        }

        return;
      }
    };

    _this.onMaskClick = function (e) {
      if (!_this.props.maskClosable) {
        return;
      }

      _this.close(e);
    };

    _this.push = function () {
      _this.setState({
        push: true
      });
    };

    _this.pull = function () {
      _this.setState({
        push: false
      });
    };

    _this.onDestroyTransitionEnd = function () {
      var isDestroyOnClose = _this.getDestroyOnClose();

      if (!isDestroyOnClose) {
        return;
      }

      if (!_this.props.visible) {
        _this.destroyClose = true;

        _this.forceUpdate();
      }
    };

    _this.getDestroyOnClose = function () {
      return _this.props.destroyOnClose && !_this.props.visible;
    }; // get drawar push width or height


    _this.getPushTransform = function (placement) {
      if (placement === 'left' || placement === 'right') {
        return "translateX(".concat(placement === 'left' ? 180 : -180, "px)");
      }

      if (placement === 'top' || placement === 'bottom') {
        return "translateY(".concat(placement === 'top' ? 180 : -180, "px)");
      }
    };

    _this.getRcDrawerStyle = function () {
      var _this$props = _this.props,
          zIndex = _this$props.zIndex,
          placement = _this$props.placement,
          style = _this$props.style;
      var push = _this.state.push;
      return _extends({
        zIndex: zIndex,
        transform: push ? _this.getPushTransform(placement) : undefined
      }, style);
    }; // render drawer body dom


    _this.renderBody = function () {
      var _this$props2 = _this.props,
          bodyStyle = _this$props2.bodyStyle,
          placement = _this$props2.placement,
          prefixCls = _this$props2.prefixCls,
          visible = _this$props2.visible;

      if (_this.destroyClose && !visible) {
        return null;
      }

      _this.destroyClose = false;
      var containerStyle = placement === 'left' || placement === 'right' ? {
        overflow: 'auto',
        height: '100%'
      } : {};

      var isDestroyOnClose = _this.getDestroyOnClose();

      if (isDestroyOnClose) {
        // Increase the opacity transition, delete children after closing.
        containerStyle.opacity = 0;
        containerStyle.transition = 'opacity .3s';
      }

      return React.createElement("div", {
        className: "".concat(prefixCls, "-wrapper-body"),
        style: containerStyle,
        onTransitionEnd: _this.onDestroyTransitionEnd
      }, _this.renderHeader(), React.createElement("div", {
        className: "".concat(prefixCls, "-body"),
        style: bodyStyle
      }, _this.props.children));
    }; // render Provider for Multi-level drawe


    _this.renderProvider = function (value) {
      var _a = _this.props,
          prefixCls = _a.prefixCls,
          zIndex = _a.zIndex,
          style = _a.style,
          placement = _a.placement,
          className = _a.className,
          wrapClassName = _a.wrapClassName,
          width = _a.width,
          height = _a.height,
          rest = __rest(_a, ["prefixCls", "zIndex", "style", "placement", "className", "wrapClassName", "width", "height"]);

      (0, _warning["default"])(wrapClassName === undefined, 'Drawer', 'wrapClassName is deprecated, please use className instead.');
      var haveMask = rest.mask ? '' : 'no-mask';
      _this.parentDrawer = value;
      var offsetStyle = {};

      if (placement === 'left' || placement === 'right') {
        offsetStyle.width = width;
      } else {
        offsetStyle.height = height;
      }

      return React.createElement(DrawerContext.Provider, {
        value: _assertThisInitialized(_this)
      }, React.createElement(_rcDrawer["default"], _extends({
        handler: false
      }, rest, offsetStyle, {
        prefixCls: prefixCls,
        open: _this.props.visible,
        onMaskClick: _this.onMaskClick,
        showMask: _this.props.mask,
        placement: placement,
        style: _this.getRcDrawerStyle(),
        className: (0, _classnames["default"])(wrapClassName, className, haveMask)
      }), _this.renderBody()));
    };

    return _this;
  }

  _createClass(Drawer, [{
    key: "componentDidUpdate",
    value: function componentDidUpdate(preProps) {
      if (preProps.visible !== this.props.visible && this.parentDrawer) {
        if (this.props.visible) {
          this.parentDrawer.push();
        } else {
          this.parentDrawer.pull();
        }
      }
    }
  }, {
    key: "renderHeader",
    value: function renderHeader() {
      var _this$props3 = this.props,
          title = _this$props3.title,
          prefixCls = _this$props3.prefixCls,
          closable = _this$props3.closable;

      if (!title && !closable) {
        return null;
      }

      var headerClassName = title ? "".concat(prefixCls, "-header") : "".concat(prefixCls, "-header-no-title");
      return React.createElement("div", {
        className: headerClassName
      }, title && React.createElement("div", {
        className: "".concat(prefixCls, "-title")
      }, title), closable && this.renderCloseIcon());
    }
  }, {
    key: "renderCloseIcon",
    value: function renderCloseIcon() {
      var _this$props4 = this.props,
          closable = _this$props4.closable,
          prefixCls = _this$props4.prefixCls;
      return closable && React.createElement("button", {
        onClick: this.close,
        "aria-label": "Close",
        className: "".concat(prefixCls, "-close")
      }, React.createElement(_icon["default"], {
        type: "close"
      }));
    }
  }, {
    key: "render",
    value: function render() {
      return React.createElement(DrawerContext.Consumer, null, this.renderProvider);
    }
  }]);

  return Drawer;
}(React.Component);

Drawer.propTypes = {
  closable: PropTypes.bool,
  destroyOnClose: PropTypes.bool,
  getContainer: PropTypes.oneOfType([PropTypes.string, PropTypes.object, PropTypes.func, PropTypes.bool]),
  maskClosable: PropTypes.bool,
  mask: PropTypes.bool,
  maskStyle: PropTypes.object,
  style: PropTypes.object,
  title: PropTypes.node,
  visible: PropTypes.bool,
  width: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
  zIndex: PropTypes.number,
  prefixCls: PropTypes.string,
  placement: PropTypes.oneOf(PlacementTypes),
  onClose: PropTypes.func,
  afterVisibleChange: PropTypes.func,
  className: PropTypes.string
};
Drawer.defaultProps = {
  width: 256,
  height: 256,
  closable: true,
  placement: 'right',
  maskClosable: true,
  mask: true,
  level: null
};

var _default = (0, _configProvider.withConfigConsumer)({
  prefixCls: 'drawer'
})(Drawer);

exports["default"] = _default;
module.exports = exports.default;
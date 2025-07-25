import * as React from 'react';
import * as PropTypes from 'prop-types';
import List, { TransferListProps } from './list';
import Operation from './operation';
import Search from './search';
export { TransferListProps } from './list';
export { TransferOperationProps } from './operation';
export { TransferSearchProps } from './search';
export declare type TransferDirection = 'left' | 'right';
declare type TransferRender = (record: TransferItem) => React.ReactNode;
export interface TransferItem {
    key: string;
    title: string;
    description?: string;
    disabled?: boolean;
}
export interface TransferProps {
    prefixCls?: string;
    className?: string;
    disabled?: boolean;
    dataSource: TransferItem[];
    targetKeys?: string[];
    selectedKeys?: string[];
    render?: TransferRender;
    onChange?: (targetKeys: string[], direction: string, moveKeys: any) => void;
    onSelectChange?: (sourceSelectedKeys: string[], targetSelectedKeys: string[]) => void;
    style?: React.CSSProperties;
    listStyle?: React.CSSProperties;
    operationStyle?: React.CSSProperties;
    titles?: string[];
    operations?: string[];
    showSearch?: boolean;
    filterOption?: (inputValue: any, item: any) => boolean;
    searchPlaceholder?: string;
    notFoundContent?: React.ReactNode;
    locale?: {};
    footer?: (props: TransferListProps) => React.ReactNode;
    body?: (props: TransferListProps) => React.ReactNode;
    rowKey?: (record: TransferItem) => string;
    onSearchChange?: (direction: TransferDirection, e: React.ChangeEvent<HTMLInputElement>) => void;
    onSearch?: (direction: TransferDirection, value: string) => void;
    lazy?: {} | boolean;
    onScroll?: (direction: TransferDirection, e: React.SyntheticEvent<HTMLDivElement>) => void;
}
export interface TransferLocale {
    titles: string[];
    notFoundContent: string;
    searchPlaceholder: string;
    itemUnit: string;
    itemsUnit: string;
}
declare class Transfer extends React.Component<TransferProps, any> {
    static List: typeof List;
    static Operation: typeof Operation;
    static Search: typeof Search;
    static defaultProps: {
        dataSource: never[];
        render: TransferRender;
        locale: {};
        showSearch: boolean;
    };
    static propTypes: {
        prefixCls: PropTypes.Requireable<string>;
        disabled: PropTypes.Requireable<boolean>;
        dataSource: PropTypes.Validator<TransferItem[]>;
        render: PropTypes.Requireable<(...args: any[]) => any>;
        targetKeys: PropTypes.Requireable<any[]>;
        onChange: PropTypes.Requireable<(...args: any[]) => any>;
        height: PropTypes.Requireable<number>;
        style: PropTypes.Requireable<object>;
        listStyle: PropTypes.Requireable<object>;
        operationStyle: PropTypes.Requireable<object>;
        className: PropTypes.Requireable<string>;
        titles: PropTypes.Requireable<any[]>;
        operations: PropTypes.Requireable<any[]>;
        showSearch: PropTypes.Requireable<boolean>;
        filterOption: PropTypes.Requireable<(...args: any[]) => any>;
        searchPlaceholder: PropTypes.Requireable<string>;
        notFoundContent: PropTypes.Requireable<PropTypes.ReactNodeLike>;
        locale: PropTypes.Requireable<object>;
        body: PropTypes.Requireable<(...args: any[]) => any>;
        footer: PropTypes.Requireable<(...args: any[]) => any>;
        rowKey: PropTypes.Requireable<(...args: any[]) => any>;
        lazy: PropTypes.Requireable<boolean | object>;
    };
    static getDerivedStateFromProps(nextProps: TransferProps): {
        sourceSelectedKeys: string[];
        targetSelectedKeys: string[];
    } | null;
    separatedDataSource: {
        leftDataSource: TransferItem[];
        rightDataSource: TransferItem[];
    } | null;
    constructor(props: TransferProps);
    separateDataSource(props: TransferProps): {
        leftDataSource: TransferItem[];
        rightDataSource: TransferItem[];
    };
    moveTo: (direction: import("../collapse/Collapse").ExpandIconPosition) => void;
    moveToLeft: () => void;
    moveToRight: () => void;
    handleSelectChange(direction: TransferDirection, holder: string[]): void;
    handleSelectAll: (direction: import("../collapse/Collapse").ExpandIconPosition, filteredDataSource: TransferItem[], checkAll: boolean) => void;
    handleLeftSelectAll: (filteredDataSource: TransferItem[], checkAll: boolean) => void;
    handleRightSelectAll: (filteredDataSource: TransferItem[], checkAll: boolean) => void;
    handleFilter: (direction: import("../collapse/Collapse").ExpandIconPosition, e: React.ChangeEvent<HTMLInputElement>) => void;
    handleLeftFilter: (e: React.ChangeEvent<HTMLInputElement>) => void;
    handleRightFilter: (e: React.ChangeEvent<HTMLInputElement>) => void;
    handleClear: (direction: import("../collapse/Collapse").ExpandIconPosition) => void;
    handleLeftClear: () => void;
    handleRightClear: () => void;
    handleSelect: (direction: import("../collapse/Collapse").ExpandIconPosition, selectedItem: TransferItem, checked: boolean) => void;
    handleLeftSelect: (selectedItem: TransferItem, checked: boolean) => void;
    handleRightSelect: (selectedItem: TransferItem, checked: boolean) => void;
    handleScroll: (direction: import("../collapse/Collapse").ExpandIconPosition, e: React.SyntheticEvent<HTMLDivElement, Event>) => void;
    handleLeftScroll: (e: React.SyntheticEvent<HTMLDivElement, Event>) => void;
    handleRightScroll: (e: React.SyntheticEvent<HTMLDivElement, Event>) => void;
    getTitles(transferLocale: TransferLocale): string[];
    getSelectedKeysName(direction: TransferDirection): "sourceSelectedKeys" | "targetSelectedKeys";
    getLocale: (transferLocale: TransferLocale, renderEmpty: (componentName?: string | undefined) => React.ReactNode) => {
        notFoundContent: any;
        searchPlaceholder: string;
        titles: string[];
        itemUnit: string;
        itemsUnit: string;
    } | {
        notFoundContent: any;
        searchPlaceholder: string;
        titles: string[];
        itemUnit: string;
        itemsUnit: string;
    };
    renderTransfer: (transferLocale: TransferLocale) => JSX.Element;
    render(): JSX.Element;
}
export default Transfer;

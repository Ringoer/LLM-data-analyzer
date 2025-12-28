#!/usr/bin/env python3
"""
GitHub仓库数据分析工具 - 支持批量仓库分析
使用: python main.py --file repos.csv --months 2023.11-2025.06 --token xxxxxx
"""

import os
import sys
import json
import math
import datetime
import statistics
import argparse
import time
from collections import defaultdict, Counter
from typing import Dict, List, Any, Optional, Tuple
import requests
from dateutil.parser import parse as parse_date
import git
import tempfile
import shutil
import threading
import random
from concurrent.futures import ThreadPoolExecutor
import csv

# ==================== 配置部分 ====================
GRAPHQL_URL = "https://api.github.com/graphql"
REST_API_BASE = "https://api.github.com"

# ==================== 进度显示工具 ====================
class ProgressTracker:
    """进度跟踪器"""

    def __init__(self, total_steps: int = 100, description: str = "进度"):
        self.total_steps = total_steps or 1
        self.current_step = 0
        self.description = description
        self.start_time = time.time()
        self.last_update_time = 0

    def update(self, increment: int = 1, message: str = ""):
        """更新进度"""
        self.current_step += increment
        current_time = time.time()

        # 限制更新频率（至少0.5秒一次）
        if current_time - self.last_update_time < 0.5 and self.current_step < self.total_steps:
            return

        self.last_update_time = current_time
        percentage = min(100, (self.current_step / self.total_steps) * 100)

        # 计算预计剩余时间
        elapsed = current_time - self.start_time
        if self.current_step > 0:
            estimated_total = elapsed / (self.current_step / self.total_steps)
            remaining = estimated_total - elapsed
            time_str = f"预计剩余: {remaining:.0f}s"
        else:
            time_str = "预计剩余: 计算中..."

        # 创建进度条
        bar_length = 30
        filled_length = int(bar_length * self.current_step // self.total_steps)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)

        # 输出进度
        sys.stdout.write(f"\r{self.description}: [{bar}] {percentage:.1f}% | {time_str}")
        if message:
            sys.stdout.write(f" | {message}")
        sys.stdout.flush()

        # 如果完成，输出新行
        if self.current_step >= self.total_steps:
            sys.stdout.write(f"\n{self.description}完成! 总耗时: {elapsed:.1f}s\n")

    def complete(self):
        """标记完成"""
        self.current_step = self.total_steps
        self.update(0, "完成")

# ==================== 工具函数 ====================
def make_aware(dt: Optional[datetime.datetime]) -> Optional[datetime.datetime]:
    """确保datetime有时区信息"""
    if dt is None:
        return None
    if isinstance(dt, str):
        dt = parse_date(dt)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=datetime.timezone.utc)
    return dt

def safe_parse_date(date_str: Optional[str]) -> Optional[datetime.datetime]:
    """安全地解析日期字符串"""
    if not date_str:
        return None
    try:
        return parse_date(date_str)
    except (ValueError, TypeError):
        return None

def parse_month_range(month_str: str) -> Tuple[datetime.datetime, datetime.datetime]:
    """解析月份范围字符串，如 '2023.11-2025.06'"""
    try:
        # 分割开始和结束月份
        start_str, end_str = month_str.split('-')

        # 解析开始月份
        start_year, start_month = map(int, start_str.split('.'))
        start_date = datetime.datetime(start_year, start_month, 1, tzinfo=datetime.timezone.utc)

        # 解析结束月份，并计算下个月的第一天（闭区间）
        end_year, end_month = map(int, end_str.split('.'))
        if end_month == 12:
            end_date = datetime.datetime(end_year + 1, 1, 1, tzinfo=datetime.timezone.utc)
        else:
            end_date = datetime.datetime(end_year, end_month + 1, 1, tzinfo=datetime.timezone.utc)

        return start_date, end_date
    except Exception as e:
        print(f"错误: 无法解析月份范围 '{month_str}', 使用默认范围")
        # 默认范围: 2023-11 到 2025-06
        return (datetime.datetime(2023, 11, 1, tzinfo=datetime.timezone.utc),
                datetime.datetime(2025, 7, 1, tzinfo=datetime.timezone.utc))

# 格式化创建时间和最后更新时间为"2023m11"格式
def format_to_year_month(date_str):
    """将日期字符串格式化为'2023m11'格式"""
    if not date_str:
        return ""
    try:
        # 解析日期字符串
        from dateutil.parser import parse
        dt = parse(date_str)
        # 格式化为"2023m11"格式
        return f"{dt.year}m{dt.month:02d}"
    except:
        return date_str  # 如果解析失败，返回原字符串

def read_repo_list(file_path: str) -> List[Dict]:
    """从CSV文件读取仓库列表，支持格式: owner/name"""
    repos = []
    try:
        print(f"正在读取仓库列表文件: {file_path}")

        # 检查文件扩展名
        if not file_path.lower().endswith('.csv'):
            print(f"错误: 文件 '{file_path}' 不是CSV格式")
            sys.exit(1)

        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            lines = list(reader)

        progress = ProgressTracker(total_steps=len(lines), description="读取仓库列表")

        for i, row in enumerate(lines, 1):
            progress.update(1, f"处理第{i}行")

            repo_str = row.get('repo', '').strip()
            if not repo_str:
                print(f"警告: 第{i}行缺少repo字段: {row}")
                continue

            # 解析仓库名格式: owner/name
            if '/' in repo_str:
                parts = repo_str.split('/')
                if len(parts) == 2:
                    owner, name = parts[0].strip(), parts[1].strip()
                    if owner and name:
                        repo_info = {
                            "owner": owner,
                            "name": name,
                            "eco_id": row.get('eco_id', ''),
                            "project_id_num": row.get('project_id_num', ''),
                            "project_type": row.get('project_type', ''),
                            "created_at": None,
                            "updated_at": None
                        }
                        repos.append(repo_info)
                    else:
                        print(f"警告: 第{i}行格式错误: {repo_str}")
                else:
                    print(f"警告: 第{i}行格式错误: {repo_str}")
            else:
                print(f"警告: 第{i}行repo字段格式错误，应为owner/name格式: {repo_str}")

        progress.complete()

    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 不存在")
        sys.exit(1)
    except Exception as e:
        print(f"错误: 读取文件 '{file_path}' 失败: {e}")
        sys.exit(1)

    if not repos:
        print("错误: 文件中没有有效的仓库信息")
        sys.exit(1)

    print(f"✓ 从文件中读取到 {len(repos)} 个仓库")
    return repos

def get_repo_metadata(owner: str, name: str, headers: Dict) -> Tuple[Optional[datetime.datetime], Optional[datetime.datetime]]:
    """获取仓库的创建时间和最后更新时间"""
    try:
        url = f"{REST_API_BASE}/repos/{owner}/{name}"
        response = requests.get(url, headers=headers, timeout=30)

        if response.status_code == 200:
            repo_data = response.json()
            created_at = safe_parse_date(repo_data.get('created_at'))
            updated_at = safe_parse_date(repo_data.get('updated_at'))

            # 转换为带时区的datetime
            if created_at:
                created_at = make_aware(created_at)
            if updated_at:
                updated_at = make_aware(updated_at)

            print(f"    获取仓库元数据: 创建时间={created_at}, 最后更新时间={updated_at}")
            return created_at, updated_at
        else:
            print(f"    获取仓库元数据失败: {response.status_code}")
            return None, None

    except Exception as e:
        print(f"    获取仓库元数据时出错: {e}")
        return None, None

def is_in_time_range(dt: Optional[datetime.datetime], start_date: datetime.datetime,
                     end_date: datetime.datetime) -> bool:
    """检查时间是否在指定范围内"""
    if dt is None:
        return False
    aware_dt = make_aware(dt)
    return start_date <= aware_dt < end_date

def run_graphql_query(query: str, headers: Dict, variables: Dict = None) -> Dict:
    """执行GraphQL查询"""
    payload = {"query": query}
    if variables:
        payload["variables"] = variables

    try:
        response = requests.post(GRAPHQL_URL, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"GraphQL查询失败: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"响应内容: {e.response.text[:200]}")
        return {"data": None, "errors": [str(e)]}

def paginate_graphql(query_template: str, node_path: str, headers: Dict,
                    batch_size: int = 100, progress_desc: str = "获取数据") -> List[Dict]:
    """分页获取所有GraphQL数据"""
    all_nodes = []
    cursor = None
    retry_count = 0
    max_retries = 3

    print(f"  {progress_desc}...")

    # 先获取第一页以了解大致数量
    first_result = run_graphql_query(query_template, headers, {"first": 1, "cursor": None})
    total_estimate = 0
    has_total_count = False

    if first_result.get("data"):
        # 尝试获取总数（如果查询包含总数）
        data = first_result.get("data", {})
        if "repository" in data:
            repo_data = data["repository"]
            for key in ["issues", "pullRequests"]:
                if key in repo_data:
                    if "totalCount" in repo_data[key]:
                        total_estimate = repo_data[key]["totalCount"]
                        has_total_count = True
                        break

    if has_total_count:
        # 有总数信息，创建正常进度条
        progress = ProgressTracker(total_steps=total_estimate // batch_size + 1,
                                 description=f"  {progress_desc}")
    else:
        # 没有总数信息，创建特殊进度显示器
        class IndeterminateProgress:
            """不确定进度的显示器"""
            def __init__(self, description):
                self.description = description
                self.count = 0
                self.start_time = time.time()
                self.last_update_time = 0

            def update(self, increment=1, message=""):
                self.count += increment
                current_time = time.time()

                # 限制更新频率（至少0.5秒一次）
                if current_time - self.last_update_time < 0.5:
                    return

                self.last_update_time = current_time
                elapsed = current_time - self.start_time

                # 只显示已获取数量，不显示百分比
                sys.stdout.write(f"\r{self.description}: 已获取 {self.count} 条")
                if message:
                    sys.stdout.write(f" | {message}")
                sys.stdout.flush()

            def complete(self):
                elapsed = time.time() - self.start_time
                sys.stdout.write(f"\n{self.description}完成! 共获取 {self.count} 条，耗时: {elapsed:.1f}s\n")
                sys.stdout.flush()

        progress = IndeterminateProgress(f"  {progress_desc}")

    total_pages = 0  # 记录实际处理的页数

    while True:
        total_pages += 1
        variables = {"first": batch_size, "cursor": cursor}
        result = run_graphql_query(query_template, headers, variables)

        # 检查错误
        if result.get("errors"):
            print(f"GraphQL错误: {result['errors']}")
            if retry_count < max_retries:
                retry_count += 1
                wait_time = 2 ** retry_count
                print(f"  等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
                continue
            else:
                print(f"重试次数超过限制 ({max_retries})，停止分页")
                break

        # 检查是否有数据
        if not result.get("data"):
            print("没有获取到数据")
            break

        # 安全地获取节点数据
        data = result.get("data", {})

        # 按照路径获取节点
        current_node = data
        path_parts = node_path.split(".")

        try:
            for part in path_parts:
                if part in current_node:
                    current_node = current_node[part]
                else:
                    current_node = None
                    break
        except AttributeError:
            current_node = None

        if not current_node:
            break

        # 处理分页结构
        if "edges" in current_node:
            edges = current_node["edges"]
            if not edges:
                break

            for edge in edges:
                if "node" in edge:
                    all_nodes.append(edge["node"])

            # 更新进度
            if has_total_count:
                # 有总数信息，显示百分比和预计时间
                progress.update(1, f"已获取 {len(all_nodes)} 条")
            else:
                # 没有总数信息，只显示数量
                progress.update(len(edges), f"第{total_pages}页")

            # 检查是否有更多数据
            page_info = current_node.get("pageInfo", {})
            if not page_info.get("hasNextPage", False):
                break

            cursor = page_info.get("endCursor")
        else:
            # 如果不是分页结构，直接添加
            if isinstance(current_node, list):
                all_nodes.extend(current_node)
            break

        retry_count = 0  # 重置重试计数

        # 添加延迟避免速率限制
        time.sleep(0.5)

    # 完成进度显示
    progress.complete()

    # 如果有总数信息但实际获取数量与估计不符，给出提示
    if has_total_count and abs(len(all_nodes) - total_estimate) > 100:
        print(f"  注意: 估计总数 {total_estimate}，实际获取 {len(all_nodes)} 条")

    print(f"  ✓ 获取到 {len(all_nodes)} 条数据")
    return all_nodes

def check_rate_limit(headers: Dict) -> bool:
    """检查API速率限制"""
    try:
        print("检查API速率限制...")
        url = f"{REST_API_BASE}/rate_limit"
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            core = data.get("resources", {}).get("core", {})
            remaining = core.get("remaining", 0)
            limit = core.get("limit", 5000)
            reset_time = datetime.datetime.fromtimestamp(core.get("reset", 0))

            print(f"✓ API速率限制: {remaining}/{limit}, 重置时间: {reset_time}")

            if remaining < 100:
                print(f"警告: API请求剩余不足: {remaining}")
                return False
            return True
    except Exception as e:
        print(f"检查速率限制失败: {e}")

    return True

# ==================== 数据收集函数 ====================
def collect_pr_data(owner: str, name: str, headers: Dict,
                   start_date: datetime.datetime, end_date: datetime.datetime) -> List[Dict]:
    """收集PR数据 - 使用REST API + 智能抽样方案，避免GraphQL 502错误"""
    print(f"  正在收集PR数据...")

    all_prs = []

    # 计算总月数用于进度跟踪
    total_months = 0
    current_month = start_date
    while current_month < end_date:
        total_months += 1
        if current_month.month == 12:
            current_month = current_month.replace(year=current_month.year + 1, month=1)
        else:
            current_month = current_month.replace(month=current_month.month + 1)

    # 重置current_month
    current_month = start_date
    month_progress = ProgressTracker(total_steps=total_months, description="    按月处理")
    month_index = 0

    while current_month < end_date:
        month_index += 1
        # 计算月份结束时间
        if current_month.month == 12:
            next_month = current_month.replace(year=current_month.year + 1, month=1)
        else:
            next_month = current_month.replace(month=current_month.month + 1)

        month_end = min(next_month, end_date)

        month_str = current_month.strftime("%Y-%m")
        month_progress.update(1, f"处理 {month_str}")

        # 使用GitHub搜索API获取该月PR列表
        month_prs = []
        page = 1
        max_pages = 10  # 每月最多1000条PR（10页×100条）

        while page <= max_pages:
            try:
                # 构建搜索查询
                search_url = f"{REST_API_BASE}/search/issues"
                created_range = f"{current_month.strftime('%Y-%m-%d')}..{month_end.strftime('%Y-%m-%d')}"
                query = f"repo:{owner}/{name} type:pr created:{created_range}"

                params = {
                    "q": query,
                    "per_page": 100,
                    "page": page,
                    "sort": "created",
                    "order": "asc"
                }

                response = requests.get(search_url, headers=headers, params=params, timeout=30)

                if response.status_code == 200:
                    data = response.json()
                    items = data.get("items", [])

                    if not items:
                        break

                    # 提取基础信息
                    for item in items:
                        pr_info = {
                            "number": item.get("number"),
                            "title": item.get("title", ""),
                            "state": item.get("state", ""),
                            "createdAt": safe_parse_date(item.get("created_at")),
                            "closedAt": safe_parse_date(item.get("closed_at")),
                            # 基础字段，后续通过抽样补充详细数据
                            "merged": None,  # 需要额外API调用获取
                            "author": {"login": item.get("user", {}).get("login")} if item.get("user") else None,
                            "comments": {"totalCount": item.get("comments", 0)},
                            "reactions": {"totalCount": 0},  # 需要额外API调用获取
                            "participants": {"totalCount": 0},  # 需要抽样获取
                            "additions": 0,
                            "deletions": 0
                        }

                        # 确保时间在范围内
                        if pr_info["createdAt"] and is_in_time_range(pr_info["createdAt"], start_date, end_date):
                            month_prs.append(pr_info)

                    # 检查是否还有更多数据
                    if len(items) < 100:
                        break

                    page += 1

                    # 添加延迟避免速率限制
                    time.sleep(0.5)

                elif response.status_code == 422:
                    # 可能是时间范围无效，尝试更小的时间范围
                    # 降级为按周获取
                    month_prs = _get_prs_by_week(owner, name, headers, current_month, month_end, start_date, end_date)
                    break

                elif response.status_code == 403 or response.status_code == 429:
                    # 速率限制
                    remaining = int(response.headers.get("X-RateLimit-Remaining", 0))
                    reset_time = datetime.datetime.fromtimestamp(
                        int(response.headers.get("X-RateLimit-Reset", 0))
                    )
                    print(f"    警告: 速率限制，剩余 {remaining} 次请求，重置时间: {reset_time}")
                    time.sleep(60)  # 等待1分钟
                    continue

                else:
                    break

            except requests.exceptions.RequestException as e:
                print(f"    网络错误: {e}")
                time.sleep(5)
                continue

        # 对该月PR进行抽样获取详细数据
        if month_prs:
            sampled_prs = _sample_and_enrich_prs(owner, name, headers, month_prs, month_str)
            all_prs.extend(sampled_prs)

        # 移动到下个月
        current_month = next_month

    month_progress.complete()
    print(f"  ✓ 收集到 {len(all_prs)} 个PR（在时间范围内）")
    return all_prs

def _get_prs_by_week(owner: str, name: str, headers: Dict,
                    start: datetime.datetime, end: datetime.datetime,
                    global_start: datetime.datetime, global_end: datetime.datetime) -> List[Dict]:
    """按周获取PR，用于处理搜索API限制的情况"""
    week_prs = []
    current_week = start

    # 计算周数
    total_weeks = ((end - start).days // 7) + 1
    week_progress = ProgressTracker(total_steps=total_weeks, description="      按周获取")

    while current_week < end:
        week_end = min(current_week + datetime.timedelta(days=7), end)
        week_progress.update(1)

        try:
            search_url = f"{REST_API_BASE}/search/issues"
            created_range = f"{current_week.strftime('%Y-%m-%d')}..{week_end.strftime('%Y-%m-%d')}"
            query = f"repo:{owner}/{name} type:pr created:{created_range}"

            params = {
                "q": query,
                "per_page": 100,
                "page": 1,
                "sort": "created",
                "order": "asc"
            }

            response = requests.get(search_url, headers=headers, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()
                items = data.get("items", [])

                for item in items:
                    pr_info = {
                        "number": item.get("number"),
                        "title": item.get("title", ""),
                        "state": item.get("state", ""),
                        "createdAt": safe_parse_date(item.get("created_at")),
                        "closedAt": safe_parse_date(item.get("closed_at")),
                        "merged": None,
                        "author": {"login": item.get("user", {}).get("login")} if item.get("user") else None,
                        "comments": {"totalCount": item.get("comments", 0)},
                        "reactions": {"totalCount": 0},
                        "participants": {"totalCount": 0},
                        "additions": 0,
                        "deletions": 0
                    }

                    if pr_info["createdAt"] and is_in_time_range(pr_info["createdAt"], global_start, global_end):
                        week_prs.append(pr_info)

            time.sleep(0.3)  # 更小的延迟

        except Exception as e:
            print(f"      按周获取失败: {e}")

        current_week = week_end

    week_progress.complete()
    return week_prs

def _sample_and_enrich_prs(owner: str, name: str, headers: Dict,
                          pr_list: List[Dict], month_str: str) -> List[Dict]:
    """对PR列表进行智能抽样并丰富数据"""
    if not pr_list:
        return []

    # 确定抽样策略
    total_prs = len(pr_list)

    if total_prs <= 50:
        # 小数量：全量获取
        sample_size = total_prs
        sampled_indices = list(range(total_prs))
    elif total_prs <= 500:
        # 中等数量：抽样40%
        sample_size = max(50, int(total_prs * 0.4))
        sample_size = min(sample_size, 200)  # 不超过200个
        sampled_indices = random.sample(range(total_prs), sample_size)
    else:
        # 大数量：固定抽样数量 + 分层抽样
        sample_size = min(500, total_prs // 20 + 200)  # 5% + 200，最多500个

        # 按状态分层抽样
        open_prs = [i for i, pr in enumerate(pr_list) if pr.get("state") == "open"]
        closed_prs = [i for i, pr in enumerate(pr_list) if pr.get("state") == "closed"]

        sampled_indices = []
        if open_prs:
            sample_open = min(5, len(open_prs) * sample_size // total_prs + 1)
            sampled_indices.extend(random.sample(open_prs, min(sample_open, len(open_prs))))

        if closed_prs:
            remaining_sample = sample_size - len(sampled_indices)
            sampled_indices.extend(random.sample(closed_prs, min(remaining_sample, len(closed_prs))))

    print(f"    从 {total_prs} 个PR中抽样 {len(sampled_indices)} 个获取详细数据")

    progress = ProgressTracker(total_steps=len(sampled_indices), description="    抽样获取详情")

    # 并发获取抽样PR的详细信息（限制并发数）
    enriched_prs = []
    semaphore = threading.Semaphore(3)  # 限制3个并发

    def enrich_pr(index):
        with semaphore:
            pr = pr_list[index]
            pr_number = pr["number"]

            try:
                # 获取PR详细信息
                pr_url = f"{REST_API_BASE}/repos/{owner}/{name}/pulls/{pr_number}"
                pr_response = requests.get(pr_url, headers=headers, timeout=15)

                if pr_response.status_code == 200:
                    pr_details = pr_response.json()

                    # 更新PR信息
                    pr["merged"] = pr_details.get("merged", False)
                    pr["additions"] = pr_details.get("additions", 0)
                    pr["deletions"] = pr_details.get("deletions", 0)

                    # 获取reactions
                    reactions_url = f"{REST_API_BASE}/repos/{owner}/{name}/issues/{pr_number}/reactions"
                    reactions_response = requests.get(reactions_url, headers=headers, timeout=15)
                    if reactions_response.status_code == 200:
                        pr["reactions"]["totalCount"] = len(reactions_response.json())

                    # 估算participants（通过评论者数量）
                    if pr["comments"]["totalCount"] > 0:
                        # 获取部分评论来估算参与者
                        comments_url = f"{REST_API_BASE}/repos/{owner}/{name}/issues/{pr_number}/comments"
                        comments_response = requests.get(comments_url, headers=headers,
                                                        params={"per_page": 10}, timeout=15)
                        if comments_response.status_code == 200:
                            comments = comments_response.json()
                            participants = set()
                            if pr.get("author") and pr["author"].get("login"):
                                participants.add(pr["author"]["login"])
                            for comment in comments[:10]:  # 只检查前10个评论
                                user = comment.get("user")
                                if user and user.get("login"):
                                    participants.add(user["login"])
                            pr["participants"]["totalCount"] = len(participants)

                return pr

            except Exception as e:
                print(f"      丰富PR #{pr_number} 数据失败: {e}")
                return pr  # 返回基础数据

    # 使用线程池并发获取
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(enrich_pr, idx) for idx in sampled_indices]
        for i, future in enumerate(futures, 1):
            try:
                enriched_pr = future.result(timeout=20)
                enriched_prs.append(enriched_pr)
                progress.update(1, f"PR #{enriched_pr.get('number', '?')}")
            except Exception as e:
                print(f"      获取PR详情超时: {e}")

    progress.complete()

    # 对于未抽样的PR，使用抽样PR的平均值估算
    if total_prs > len(sampled_indices) and enriched_prs:
        # 计算抽样PR的平均值
        avg_reactions = sum(p["reactions"]["totalCount"] for p in enriched_prs) / len(enriched_prs)
        avg_participants = sum(p["participants"]["totalCount"] for p in enriched_prs) / len(enriched_prs)

        # 为未抽样的PR应用估算值
        print(f"    为剩余 {total_prs - len(sampled_indices)} 个PR应用估算值...")
        estimate_progress = ProgressTracker(total_steps=total_prs - len(sampled_indices), description="    估算数据")

        for i, pr in enumerate(pr_list):
            if i not in sampled_indices:
                pr["reactions"]["totalCount"] = int(avg_reactions)
                pr["participants"]["totalCount"] = int(avg_participants)

                # 根据状态估算merged
                if pr["state"] == "closed":
                    # 计算抽样PR中closed状态的合并率
                    closed_sampled = [p for p in enriched_prs if p.get("state") == "closed"]
                    if closed_sampled:
                        merge_rate = sum(1 for p in closed_sampled if p.get("merged")) / len(closed_sampled)
                        pr["merged"] = random.random() < merge_rate  # 按概率设置
                else:
                    pr["merged"] = False

                enriched_prs.append(pr)
                estimate_progress.update(1)

        estimate_progress.complete()

    print(f"    ✓ {month_str} 完成PR数据丰富")
    return enriched_prs


def collect_issue_data(owner: str, name: str, headers: Dict,
                      start_date: datetime.datetime, end_date: datetime.datetime) -> List[Dict]:
    """收集issue数据"""
    print(f"  正在收集issue数据...")

    issue_query = """
query($first: Int!, $cursor: String) {
  repository(owner: "%s", name: "%s") {
    issues(first: $first, after: $cursor, orderBy: {field: CREATED_AT, direction: ASC}) {
      edges {
        node {
          number
          title
          state
          createdAt
          closedAt
          comments(first: 1) {
            totalCount
          }
          reactions(first: 1) {
            totalCount
          }
        }
      }
      pageInfo {
        hasNextPage
        endCursor
      }
    }
  }
}
""" % (owner, name)

    try:
        issues = paginate_graphql(issue_query, "repository.issues", headers, progress_desc="获取issue数据")

        print("  过滤时间范围内的issue...")
        progress = ProgressTracker(total_steps=len(issues), description="  时间过滤")
        filtered_issues = []

        for i, issue in enumerate(issues):
            progress.update(1, f"处理第{i+1}个issue")

            if not isinstance(issue, dict):
                continue

            created_at_str = issue.get("createdAt")
            if not created_at_str:
                continue

            created_at = safe_parse_date(created_at_str)
            if created_at and is_in_time_range(created_at, start_date, end_date):
                issue["createdAt"] = make_aware(created_at)

                closed_at_str = issue.get("closedAt")
                if closed_at_str:
                    closed_at = safe_parse_date(closed_at_str)
                    issue["closedAt"] = make_aware(closed_at) if closed_at else None

                filtered_issues.append(issue)

        progress.complete()
        print(f"  ✓ 收集到 {len(filtered_issues)} 个issue（在时间范围内）")
        return filtered_issues

    except Exception as e:
        print(f"  收集issue数据时出错: {e}")
        return []

def get_pr_commits_sample_monthly(owner: str, name: str, monthly_pr_numbers: Dict[str, List[int]],
                                 headers: Dict, max_sample_per_month: int = 30) -> Dict[str, Dict[int, int]]:
    """按月抽样获取PR的提交作者数（用于统计多作者PR）"""
    print("  按月抽样获取PR提交作者数...")

    monthly_pr_authors = {}
    total_prs_to_sample = sum(min(max_sample_per_month, len(prs)) for prs in monthly_pr_numbers.values())

    if total_prs_to_sample == 0:
        return {}

    progress = ProgressTracker(total_steps=total_prs_to_sample, description="  抽样PR提交作者")

    for month_key, pr_numbers in monthly_pr_numbers.items():
        if not pr_numbers or max_sample_per_month <= 0:
            monthly_pr_authors[month_key] = {}
            continue

        # 计算抽样间隔
        sample_step = max(1, len(pr_numbers) // min(max_sample_per_month, len(pr_numbers)))
        sampled_prs = pr_numbers[::sample_step][:max_sample_per_month]

        pr_authors_count = {}

        for pr_num in sampled_prs:
            try:
                # 使用REST API获取PR的commits
                url = f"{REST_API_BASE}/repos/{owner}/{name}/pulls/{pr_num}/commits"
                response = requests.get(url, headers=headers, params={"per_page": 100}, timeout=30)

                if response.status_code == 200:
                    commits = response.json()
                    authors = set()
                    for commit in commits:
                        if isinstance(commit, dict):
                            if commit.get("author") and isinstance(commit["author"], dict):
                                author_login = commit["author"].get("login")
                                if author_login:
                                    authors.add(author_login)

                            commit_info = commit.get("commit", {})
                            if isinstance(commit_info, dict):
                                author_info = commit_info.get("author", {})
                                if isinstance(author_info, dict):
                                    author_email = author_info.get("email")
                                    if author_email:
                                        authors.add(author_email)

                    pr_authors_count[pr_num] = len(authors)
                    progress.update(1, f"PR #{pr_num}")
                elif response.status_code == 404:
                    print(f"  警告: PR #{pr_num} 不存在或已删除")
                elif response.status_code == 410:
                    print(f"  警告: PR #{pr_num} 已被关闭")
                else:
                    print(f"  警告: 获取PR #{pr_num}的commits失败: {response.status_code}")

                # 遵守速率限制
                if "X-RateLimit-Remaining" in response.headers:
                    remaining = int(response.headers["X-RateLimit-Remaining"])
                    if remaining < 10:
                        print(f"  警告: API速率限制即将用完，剩余 {remaining}")
                        time.sleep(5)  # 暂停一下

            except Exception as e:
                print(f"  获取PR #{pr_num}的commits失败: {e}")

        monthly_pr_authors[month_key] = pr_authors_count

    progress.complete()
    print(f"  ✓ 完成 {total_prs_to_sample} 个PR的提交作者抽样")
    return monthly_pr_authors

def calculate_rfc_issue_adoption_rate_ultra_simple(owner: str, name: str, headers: Dict,
                                                 start_date: datetime.datetime, end_date: datetime.datetime) -> Dict:
    """极简版RFC issue采纳率计算 - 按月统计"""

    print("  正在计算RFC issue采纳率...")

    # 初始化月度数据
    monthly_stats = {}
    current = start_date
    while current < end_date:
        month_key = current.strftime("%Y-%m")
        monthly_stats[month_key] = {
            "total_rfc_issues": 0,
            "issues_with_direct_pr": 0,
            "merged_related_prs": 0,
            "rfc_issue_adoption_rate": 0
        }
        # 移动到下个月
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)

    # 搜索所有RFC issue
    search_url = f"{REST_API_BASE}/search/issues"
    date_range = f"{start_date.strftime('%Y-%m-%d')}..{end_date.strftime('%Y-%m-%d')}"
    query = f"repo:{owner}/{name} RFC in:title type:issue created:{date_range}"

    print("  搜索RFC issue...")
    params = {"q": query, "per_page": 100}
    response = requests.get(search_url, headers=headers, params=params, timeout=30)

    if response.status_code != 200:
        print(f"    搜索RFC issue失败: {response.status_code}")
        return {
            "monthly": monthly_stats,
            "overall": {
                "total_rfc_issues": 0,
                "issues_with_direct_pr": 0,
                "merged_related_prs": 0,
                "rfc_issue_adoption_rate": 0
            }
        }

    data = response.json()
    items = data.get("items", [])

    print(f"  找到 {len(items)} 个RFC issue，正在分析...")
    progress = ProgressTracker(total_steps=len(items), description="  分析RFC issue")

    # 按月统计
    for i, item in enumerate(items):
        progress.update(1, f"第{i+1}个")
        created_at = safe_parse_date(item.get("created_at"))
        if not created_at:
            continue

        month_key = created_at.strftime("%Y-%m")
        if month_key not in monthly_stats:
            continue

        monthly_stats[month_key]["total_rfc_issues"] += 1

        pr_data = item.get("pull_request")
        if pr_data and pr_data.get("merged_at") is not None:
            monthly_stats[month_key]["issues_with_direct_pr"] += 1
            if pr_data["merged_at"]:  # 有具体的合并时间戳
                monthly_stats[month_key]["merged_related_prs"] += 1

    progress.complete()

    # 计算每月的采纳率
    print("  计算月度采纳率...")
    for month_key, stats in monthly_stats.items():
        if stats["issues_with_direct_pr"] > 0:
            stats["rfc_issue_adoption_rate"] = round(
                (stats["merged_related_prs"] / stats["issues_with_direct_pr"] * 100), 2
            )

    # 计算总体统计
    total_issues = sum(stats["total_rfc_issues"] for stats in monthly_stats.values())
    total_with_pr = sum(stats["issues_with_direct_pr"] for stats in monthly_stats.values())
    total_merged = sum(stats["merged_related_prs"] for stats in monthly_stats.values())
    overall_rate = round((total_merged / total_with_pr * 100), 2) if total_with_pr > 0 else 0

    print(f"  ✓ RFC issue采纳率分析完成: 找到 {total_issues} 个RFC issue")
    return {
        "monthly": monthly_stats,
        "overall": {
            "total_rfc_issues": total_issues,
            "issues_with_direct_pr": total_with_pr,
            "merged_related_prs": total_merged,
            "rfc_issue_adoption_rate": overall_rate
        }
    }

def collect_fork_data(owner: str, name: str, headers: Dict,
                     start_date: datetime.datetime, end_date: datetime.datetime) -> List[Dict]:
    """收集fork数据"""
    print(f"  正在收集fork数据...")
    forks = []
    page = 1
    total_forks = 0

    # 先获取第一页来估计总数
    try:
        url = f"{REST_API_BASE}/repos/{owner}/{name}/forks"
        first_response = requests.get(url, headers=headers, params={
            "per_page": 1,
            "page": 1,
            "sort": "newest"
        }, timeout=30)

        if first_response.status_code == 200:
            # 从响应头获取估计的总数
            if "Link" in first_response.headers:
                # 解析Link头中的last页数
                link_header = first_response.headers["Link"]
                if "rel=\"last\"" in link_header:
                    # 简单解析，实际应该使用更健壮的解析
                    import re
                    match = re.search(r'page=(\d+)>; rel="last"', link_header)
                    if match:
                        total_pages = int(match.group(1))
                        total_forks = total_pages * 100  # 每页最多100个
    except:
        pass

    if total_forks == 0:
        total_forks = 1000  # 默认估计值

    progress = ProgressTracker(total_steps=total_forks // 100 + 1, description="  获取fork数据")

    while True:
        try:
            url = f"{REST_API_BASE}/repos/{owner}/{name}/forks"
            response = requests.get(url, headers=headers, params={
                "per_page": 100,
                "page": page,
                "sort": "newest"
            }, timeout=30)

            if response.status_code != 200:
                print(f"  获取fork数据失败: {response.status_code}")
                break

            batch_forks = response.json()
            if not batch_forks:
                break

            for fork in batch_forks:
                if isinstance(fork, dict):
                    created_at_str = fork.get("created_at")
                    if created_at_str:
                        created_at = safe_parse_date(created_at_str)
                        if created_at and is_in_time_range(created_at, start_date, end_date):
                            fork["created_at"] = make_aware(created_at)
                            forks.append(fork)

            progress.update(1, f"第{page}页，已收集{len(forks)}个")

            # 如果最早的fork已经超出时间范围，可以提前停止
            if batch_forks and len(batch_forks) == 100:
                last_fork = batch_forks[-1]
                if isinstance(last_fork, dict):
                    last_date_str = last_fork.get("created_at")
                    if last_date_str:
                        last_date = safe_parse_date(last_date_str)
                        if last_date and last_date < start_date:
                            break

            page += 1
            time.sleep(0.2)  # 添加延迟避免速率限制

        except Exception as e:
            print(f"  收集fork数据时出错: {e}")
            break

    progress.complete()
    print(f"  ✓ 收集到 {len(forks)} 个fork（在时间范围内）")
    return forks

# ==================== Git本地分析 ====================
def analyze_git_repo(owner: str, name: str, start_date: datetime.datetime,
                    end_date: datetime.datetime) -> Dict:
    """克隆并分析Git仓库"""
    print(f"  正在克隆仓库进行本地分析...")

    temp_dir = tempfile.mkdtemp(prefix=f"github_analysis_{owner}_{name}_")
    repo_url = f"https://github.com/{owner}/{name}.git"

    try:
        # 克隆仓库（浅克隆以提高速度）
        print(f"  正在克隆 {repo_url}...")

        # 创建自定义的进度类
        class CloneProgress(git.remote.RemoteProgress):
            def __init__(self):
                super().__init__()
                self.progress = ProgressTracker(total_steps=100, description="  Git克隆")

            def update(self, op_code, cur_count, max_count=None, message=''):
                if message:
                    print(f"  进度: {message}")

        progress = CloneProgress()

        print("  开始克隆...")
        repo = git.Repo.clone_from(repo_url, temp_dir, single_branch=True, progress=progress)

        progress.progress.complete()
        print("  ✓ Git克隆完成")

        print("  获取commits...")
        # 获取所有commit
        try:
            commits = list(repo.iter_commits(since=start_date.isoformat(),
                                            until=end_date.isoformat()))
            print(f"  找到 {len(commits)} 个commits")
        except git.exc.GitCommandError as e:
            print(f"  获取commits失败，尝试其他方法: {e}")
            # 如果标准方法失败，尝试使用git log命令
            from git import Git
            g = Git(temp_dir)
            log_output = g.log(f'--since="{start_date.isoformat()}"',
                              f'--until="{end_date.isoformat()}"',
                              '--pretty=format:%H')
            commit_hashes = log_output.split('\n') if log_output else []
            commits = [repo.commit(hash) for hash in commit_hashes if hash]
            print(f"  通过git log找到 {len(commits)} 个commits")
        progress = ProgressTracker(total_steps=len(commits), description="  分析commits")

        # 初始化月度数据结构
        monthly_data = {}

        # 初始化月度贡献者统计（新增）
        monthly_contributors = {}

        # 预创建所有月份的条目
        current = start_date
        while current < end_date:
            month_key = current.strftime("%Y-%m")
            monthly_data[month_key] = {
                "total_commits": 0,
                "non_merge_commits": 0,
                "author_commits": Counter(),
                "commits_list": []
            }
            # 初始化贡献者集合（新增）
            monthly_contributors[month_key] = set()
            # 移动到下个月
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)

        # 按月份分类commits并统计贡献者
        for i, commit in enumerate(commits):
            progress.update(1, f"commit {i+1}/{len(commits)}")
            try:
                commit_time = make_aware(commit.committed_datetime)
                month_key = commit_time.strftime("%Y-%m")

                # 只处理在预定义月份范围内的commit
                if month_key not in monthly_data:
                    continue

                # 1. 基本提交统计
                monthly_data[month_key]["total_commits"] += 1
                monthly_data[month_key]["commits_list"].append(commit)

                # 排除merge commit（超过1个父提交）
                if len(commit.parents) <= 1:
                    monthly_data[month_key]["non_merge_commits"] += 1

                    # 统计作者提交次数（用于top_10计算）
                    if commit.author.email:
                        author_key = commit.author.email
                    elif commit.author.name:
                        author_key = commit.author.name
                    else:
                        author_key = "unknown"

                    monthly_data[month_key]["author_commits"][author_key] += 1

                    # 2. 统计月度不重复贡献者（新增）
                    if author_key != "unknown":
                        monthly_contributors[month_key].add(author_key)

            except Exception as e:
                print(f"  处理commit {commit.hexsha[:8]} 时出错: {e}")
                continue

        progress.complete()

        # 计算月度贡献者人数（新增）
        monthly_unique_contributors = {}
        for month_key, contributors_set in monthly_contributors.items():
            monthly_unique_contributors[month_key] = len(contributors_set)

        # 计算每个月的top_10_developer_percentage和其他指标
        print("  计算月度开发者统计...")
        monthly_progress = ProgressTracker(total_steps=len(monthly_data), description="  月度统计")
        monthly_results = {}
        total_non_merge_commits = 0
        all_author_commits = Counter()
        all_contributors = set()  # 新增：总体贡献者集合

        for month_key, month_data in monthly_data.items():
            monthly_progress.update(1, f"月份 {month_key}")
            author_commits = month_data["author_commits"]
            total_commits = sum(author_commits.values())

            # 累积总体数据
            total_non_merge_commits += month_data["non_merge_commits"]
            all_author_commits.update(author_commits)

            # 累积总体贡献者（新增）
            month_contributors = monthly_contributors.get(month_key, set())
            all_contributors.update(month_contributors)

            # 计算本月top_10_developer_percentage
            if total_commits > 0 and len(author_commits) > 0:
                sorted_authors = sorted(author_commits.items(), key=lambda x: x[1], reverse=True)
                top_10_count = max(1, math.ceil(len(sorted_authors) * 0.1))
                top_10_commits = sum(count for _, count in sorted_authors[:top_10_count])
                top_10_percentage = (top_10_commits / total_commits) * 100
            else:
                top_10_percentage = 0

            monthly_results[month_key] = {
                "total_commits": month_data["total_commits"],
                "non_merge_commits": month_data["non_merge_commits"],
                "author_count": len(author_commits),
                "unique_contributors": monthly_unique_contributors.get(month_key, 0),  # 新增
                "top_10_percentage": round(top_10_percentage, 2),
                "author_commits_summary": dict(author_commits.most_common(10))  # 只保存前10作者
            }

        monthly_progress.complete()

        # 计算总体top_10_percentage
        overall_top_10_percentage = 0
        if all_author_commits:
            sorted_all_authors = sorted(all_author_commits.items(), key=lambda x: x[1], reverse=True)
            top_10_count = max(1, math.ceil(len(sorted_all_authors) * 0.1))
            top_10_commits = sum(count for _, count in sorted_all_authors[:top_10_count])
            total_all_commits = sum(all_author_commits.values())
            overall_top_10_percentage = round((top_10_commits / total_all_commits) * 100, 2)

        # 获取tags
        print("  分析tags...")
        tags = []
        tag_list = list(repo.tags)
        tag_progress = ProgressTracker(total_steps=len(tag_list), description="  分析tags")

        for tag in tag_list:
            tag_progress.update(1)
            try:
                # 获取tag的commit时间
                commit = tag.commit
                commit_time = make_aware(commit.committed_datetime)

                if is_in_time_range(commit_time, start_date, end_date):
                    tags.append({
                        "name": tag.name,
                        "date": commit_time,
                        "commit": commit.hexsha
                    })
            except Exception as e:
                print(f"  处理tag {tag.name} 时出错: {e}")

        tag_progress.complete()

        print(f"  ✓ Git分析完成")
        return {
            "monthly_commits": monthly_results,
            "monthly_unique_contributors": monthly_unique_contributors,  # 新增：单独提供贡献者数据
            "tags": tags,
            "tag_count": len(tags),
            "overall": {
                "total_commits": len(commits),
                "non_merge_commits": total_non_merge_commits,
                "top_10_percentage": overall_top_10_percentage,
                "total_authors": len(all_author_commits),
                "total_unique_contributors": len(all_contributors)  # 新增：总体贡献者人数
            }
        }

    except git.exc.GitCommandError as e:
        print(f"  Git操作失败: {e}")
        return {}
    except Exception as e:
        print(f"  Git分析失败: {e}")
        import traceback
        traceback.print_exc()
        return {}
    finally:
        # 清理临时目录
        if os.path.exists(temp_dir):
            try:
                print("  清理临时目录...")
                shutil.rmtree(temp_dir, ignore_errors=True)
            except:
                pass

# ==================== 数据分析函数 ====================
def calculate_monthly_metrics(prs: List[Dict], issues: List[Dict],
                            forks: List[Dict], git_data: Dict,
                            start_date: datetime.datetime, end_date: datetime.datetime,
                            owner: str, name: str, headers: Dict) -> Dict:
    """计算月度指标"""
    print("  正在计算月度指标...")

    # 初始化月度数据结构
    months = {}
    current = start_date
    total_months = 0

    while current < end_date:
        total_months += 1
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)

    # 重置current_month
    current = start_date
    progress = ProgressTracker(total_steps=total_months, description="  初始化月度数据")

    while current < end_date:
        month_key = current.strftime("%Y-%m")
        months[month_key] = {
            "prs": [],
            "issues": [],
            "comments": 0,
            "reactions": 0,
            "forks": [],
            "tags": [],
            "pr_merged": 0,
            "pr_total": 0,
            "rfc_prs": [],
            "issue_resolution_days": [],
            # 新增月度指标字段
            "avg_issue_resolution_days": 0,  # 指标2
            "rfc_adoption_rate": 0,          # 指标4
            "top_10_developer_commit_percentage": 0,  # 指标5
            "non_merge_commits": 0,          # 指标6
            "multi_author_pr_count": 0,      # 多作者PR次数
            "multi_author_pr_rate": 0        # 多作者PR比例
        }
        progress.update(1, f"月份 {month_key}")
        # 移动到下个月
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)

    progress.complete()

    print("  分配PR到月份...")
    pr_progress = ProgressTracker(total_steps=len(prs), description="  处理PR")
    # 分配PR到月份（按创建时间）
    for i, pr in enumerate(prs):
        pr_progress.update(1, f"PR {i+1}/{len(prs)}")
        if isinstance(pr, dict) and pr.get("createdAt"):
            created_at = pr["createdAt"]
            if isinstance(created_at, datetime.datetime):
                month_key = created_at.strftime("%Y-%m")
                if month_key in months:
                    months[month_key]["prs"].append(pr)
                    months[month_key]["pr_total"] += 1
                    if pr.get("merged"):
                        months[month_key]["pr_merged"] += 1

                    # 检查RFC PR
                    title = pr.get("title", "").lower()
                    if "RFC" in title:
                        months[month_key]["rfc_prs"].append(pr)

    pr_progress.complete()

    print("  分配issue到月份...")
    issue_progress = ProgressTracker(total_steps=len(issues), description="  处理issue")
    # 分配issue到月份并计算解决周期
    for i, issue in enumerate(issues):
        issue_progress.update(1, f"issue {i+1}/{len(issues)}")
        if isinstance(issue, dict) and issue.get("createdAt"):
            created_at = issue["createdAt"]
            if isinstance(created_at, datetime.datetime):
                month_key = created_at.strftime("%Y-%m")
                if month_key in months:
                    months[month_key]["issues"].append(issue)

                    # 计算解决周期（如果已关闭）
                    if issue.get("closedAt") and isinstance(issue["closedAt"], datetime.datetime):
                        resolution_days = (issue["closedAt"] - issue["createdAt"]).days
                        if resolution_days >= 0:  # 有效天数
                            months[month_key]["issue_resolution_days"].append(resolution_days)

    issue_progress.complete()

    print("  统计评论和反应...")
    stats_progress = ProgressTracker(total_steps=len(months), description="  统计交互数据")
    # 统计评论和反应
    for month_key, month_data in months.items():
        stats_progress.update(1, f"月份 {month_key}")
        # PR的评论和反应
        for pr in month_data["prs"]:
            if isinstance(pr, dict):
                month_data["comments"] += pr.get("comments", {}).get("totalCount", 0)
                month_data["reactions"] += pr.get("reactions", {}).get("totalCount", 0)

        # issue的评论和反应
        for issue in month_data["issues"]:
            if isinstance(issue, dict):
                month_data["comments"] += issue.get("comments", {}).get("totalCount", 0)
                month_data["reactions"] += issue.get("reactions", {}).get("totalCount", 0)

    stats_progress.complete()

    print("  分配fork到月份...")
    fork_progress = ProgressTracker(total_steps=len(forks), description="  处理fork")
    # 分配fork到月份
    for i, fork in enumerate(forks):
        fork_progress.update(1, f"fork {i+1}/{len(forks)}")
        if isinstance(fork, dict) and fork.get("created_at"):
            created_at = fork["created_at"]
            if isinstance(created_at, datetime.datetime):
                month_key = created_at.strftime("%Y-%m")
                if month_key in months:
                    months[month_key]["forks"].append(fork)

    fork_progress.complete()

    # 分配tag到月份
    if git_data and "tags" in git_data:
        print("  分配tag到月份...")
        tag_progress = ProgressTracker(total_steps=len(git_data["tags"]), description="  处理tag")
        for i, tag in enumerate(git_data["tags"]):
            tag_progress.update(1, f"tag {i+1}/{len(git_data['tags'])}")
            if isinstance(tag, dict) and tag.get("date"):
                tag_date = tag["date"]
                if isinstance(tag_date, datetime.datetime):
                    month_key = tag_date.strftime("%Y-%m")
                    if month_key in months:
                        months[month_key]["tags"].append(tag)

        tag_progress.complete()

    # 新增：计算RFC issue采纳率（按月统计）
    rfc_issue_result = calculate_rfc_issue_adoption_rate_ultra_simple(owner, name, headers, start_date, end_date)

    # 将RFC issue采纳率月度数据合并到主月度数据中
    print("  合并RFC issue数据...")
    for month_key in months:
        if month_key in rfc_issue_result["monthly"]:
            months[month_key]["rfc_issue_stats"] = rfc_issue_result["monthly"][month_key]

    print(f"  ✓ 月度指标计算完成")
    return months

def calculate_overall_metrics(prs: List[Dict], issues: List[Dict],
                            months: Dict, git_data: Dict,
                            owner: str, name: str, headers: Dict) -> Dict:
    """计算总体指标"""
    print("  正在计算总体指标...")

    # 准备按月PR编号用于多作者PR抽样
    monthly_pr_numbers = {}
    print("  准备月度PR编号...")
    month_progress = ProgressTracker(total_steps=len(months), description="  准备PR数据")
    for month_key, month_data in months.items():
        month_progress.update(1, f"月份 {month_key}")
        pr_numbers = []
        for pr in month_data["prs"]:
            if isinstance(pr, dict) and "number" in pr:
                pr_numbers.append(pr["number"])
        monthly_pr_numbers[month_key] = pr_numbers

    month_progress.complete()

    # 按月获取多作者PR数据
    monthly_pr_authors = get_pr_commits_sample_monthly(owner, name, monthly_pr_numbers, headers, max_sample_per_month=30)

    print("  计算月度指标...")
    calc_progress = ProgressTracker(total_steps=len(months), description="  月度计算")
    # 计算每个月的指标
    for month_key, month_data in months.items():
        calc_progress.update(1, f"月份 {month_key}")
        # 1. 月度issue解决周期（平均天数）- 指标2
        if month_data["issue_resolution_days"]:
            month_data["avg_issue_resolution_days"] = round(
                statistics.mean(month_data["issue_resolution_days"]), 2
            )
        else:
            month_data["avg_issue_resolution_days"] = 0

        # 2. 月度RFC采纳率 - 指标4
        # 2.1 RFC PR数据
        rfc_pr_merged = sum(1 for pr in month_data["rfc_prs"] if isinstance(pr, dict) and pr.get("merged"))
        rfc_pr_adoption_rate = (rfc_pr_merged / len(month_data["rfc_prs"]) * 100) if month_data["rfc_prs"] else 0

        # 2.2 RFC issue数据
        total_rfc_issues = 0
        merged_rfc_issues = 0
        if "rfc_issue_stats" in month_data:
            stats = month_data["rfc_issue_stats"]
            total_rfc_issues = stats.get("total_rfc_issues", 0)
            merged_rfc_issues = stats.get("merged_related_prs", 0)

        # 2.3 计算月度综合RFC采纳率
        total_merged = merged_rfc_issues + rfc_pr_merged
        total_all = total_rfc_issues + len(month_data["rfc_prs"])

        if total_all > 0:
            month_data["rfc_adoption_rate"] = round((total_merged / total_all * 100), 2)
        else:
            month_data["rfc_adoption率"] = 0

        # 3. 月度开发者集中度和提交数 - 指标5和6（使用准确的月度Git数据）
        if git_data and "monthly_commits" in git_data:
            month_commits = git_data["monthly_commits"].get(month_key)
            if month_commits:
                # 月度开发者集中度
                month_data["top_10_developer_commit_percentage"] = month_commits.get("top_10_percentage", 0)
                # 月度非merge提交数
                month_data["non_merge_commits"] = month_commits.get("non_merge_commits", 0)
                # 月度贡献者人数
                month_data["unique_contributors"] = month_commits.get("unique_contributors", 0)
            else:
                # 如果没有该月数据，设为0
                month_data["top_10_developer_commit_percentage"] = 0
                month_data["non_merge_commits"] = 0
                month_data["unique_contributors"] = 0
        else:
            # 如果没有Git数据，设为0
            month_data["top_10_developer_commit_percentage"] = 0
            month_data["non_merge_commits"] = 0
            month_data["unique_contributors"] = 0

        # 4. 月度多作者PR统计
        if month_key in monthly_pr_authors:
            pr_authors_count = monthly_pr_authors[month_key]
            if pr_authors_count:
                multi_author_prs = sum(1 for count in pr_authors_count.values() if count >= 2)
                prs = len(monthly_pr_numbers[month_key])
                multi_author_pr_rate = round((multi_author_prs / len(pr_authors_count) * 100), 2)
                multi_author_pr_count = int(multi_author_pr_rate * prs)
                month_data["multi_author_pr_count"] = multi_author_pr_count
                month_data["multi_author_pr_rate"] = multi_author_pr_rate
            else:
                month_data["multi_author_pr_count"] = 0
                month_data["multi_author_pr_rate"] = 0
        else:
            month_data["multi_author_pr_count"] = 0
            month_data["multi_author_pr_rate"] = 0

    calc_progress.complete()

    print("  计算总体指标...")
    # 计算总体指标
    # 1. PR合并率
    total_prs = sum(m["pr_total"] for m in months.values())
    merged_prs = sum(m["pr_merged"] for m in months.values())
    pr_merge_rate = round((merged_prs / total_prs * 100), 2) if total_prs > 0 else 0

    # 2. issue解决周期（平均天数）
    all_resolution_days = []
    for month_data in months.values():
        all_resolution_days.extend(month_data["issue_resolution_days"])

    avg_issue_resolution = round(statistics.mean(all_resolution_days), 2) if all_resolution_days else 0

    # 3. 每月comment和reaction总数
    monthly_interaction = {
        month: {
            "comments": data["comments"],
            "reactions": data["reactions"],
            "total": data["comments"] + data["reactions"]
        }
        for month, data in months.items()
    }

    # 4. RFC采纳率（总体）
    # 4.1 RFC PR数据
    total_rfc_prs = []
    for month_data in months.values():
        total_rfc_prs.extend(month_data["rfc_prs"])

    rfc_pr_merged = sum(1 for pr in total_rfc_prs if isinstance(pr, dict) and pr.get("merged"))
    rfc_pr_adoption_rate = round((rfc_pr_merged / len(total_rfc_prs) * 100), 2) if total_rfc_prs else 0

    # 4.2 RFC issue数据 - 从月度数据汇总
    total_rfc_issues = 0
    merged_rfc_issues = 0

    for month_data in months.values():
        if "rfc_issue_stats" in month_data:
            stats = month_data["rfc_issue_stats"]
            total_rfc_issues += stats.get("total_rfc_issues", 0)
            merged_rfc_issues += stats.get("merged_related_prs", 0)

    # 4.3 计算综合RFC采纳率 (merged rfc issue + merged rfc pr) / (total rfc issue + total rfc pr)
    total_merged = merged_rfc_issues + rfc_pr_merged
    total_all = total_rfc_issues + len(total_rfc_prs)

    rfc_adoption_rate = round((total_merged / total_all * 100), 2) if total_all else 0

    # 5. 提交量前10%的开发者占比（总体）
    if git_data and "overall" in git_data:
        top_10_percentage = git_data["overall"].get("top_10_percentage", 0)
    else:
        top_10_percentage = 0

    # 6. 代码commit数量（不包括merge）（总体）
    if git_data and "overall" in git_data:
        non_merge_commits = git_data["overall"].get("non_merge_commits", 0)
    else:
        non_merge_commits = 0

    # 7. 每月发布新tag的数量
    monthly_tags = {month: len(data["tags"]) for month, data in months.items()}

    # 8. 每月贡献者人数
    monthly_contributors = {month: data["unique_contributors"] for month, data in months.items()}

    # 9. 被fork数
    monthly_forks = {month: len(data["forks"]) for month, data in months.items()}

    # 10. PR数量（至少包含2位作者） - 基于抽样
    # 从月度数据中汇总抽样结果
    total_multi_author_prs = sum(data["multi_author_pr_count"] for data in months.values())
    total_sampled_prs = sum(len(monthly_pr_authors.get(month_key, {})) for month_key in months.keys())

    multi_author_rate = round((total_multi_author_prs / total_sampled_prs * 100), 2) if total_sampled_prs > 0 else 0

    # 汇总月度指标数据
    monthly_metrics = {
        "pr_counts": {month: data["pr_total"] for month, data in months.items()},
        "pr_merge_rates": {
            month: round((data["pr_merged"] / data["pr_total"] * 100), 2) if data["pr_total"] > 0 else 0
            for month, data in months.items()
        },
        "avg_issue_resolution_days": {month: data["avg_issue_resolution_days"] for month, data in months.items()},
        "rfc_adoption_rates": {month: data["rfc_adoption_rate"] for month, data in months.items()},
        "top_10_developer_percentages": {month: data["top_10_developer_commit_percentage"] for month, data in months.items()},
        "non_merge_commits": {month: data["non_merge_commits"] for month, data in months.items()},
        "unique_contributors": monthly_contributors,
        "multi_author_pr_counts": {month: data["multi_author_pr_count"] for month, data in months.items()},  # 多作者PR次数
        "multi_author_pr_rates": {month: data["multi_author_pr_rate"] for month, data in months.items()},   # 多作者PR比例
        "interaction": monthly_interaction,
        "tags": monthly_tags,
        "forks": monthly_forks
    }

    print(f"  ✓ 总体指标计算完成")
    return {
        "overall": {
            "pr_merge_rate": pr_merge_rate,
            "avg_issue_resolution_days": avg_issue_resolution,
            "rfc_adoption_rate": rfc_adoption_rate,
            "rfc_pr_adoption_rate": rfc_pr_adoption_rate,
            "rfc_issue_adoption_rate": round((merged_rfc_issues / total_rfc_issues * 100), 2) if total_rfc_issues > 0 else 0,
            "top_10_developer_commit_percentage": round(top_10_percentage, 2),
            "total_non_merge_commits": non_merge_commits,
            "multi_author_pr_count": total_multi_author_prs,  # 多作者PR次数
            "multi_author_pr_rate": multi_author_rate,        # 多作者PR比例,
            "total_unique_contributors": sum(monthly_contributors.values()),
            "avg_monthly_contributors": round(statistics.mean(list(monthly_contributors.values())), 2) if monthly_contributors else 0,
            "total_forks": sum(monthly_forks.values()),
            "total_tags": sum(monthly_tags.values()),
            "total_prs": total_prs,
            "total_issues": sum(len(m["issues"]) for m in months.values())
        },
        "monthly": monthly_metrics,
        "sample_stats": {
            "pr_authors_sample_size": total_sampled_prs,
            "multi_author_prs_in_sample": total_multi_author_prs
        }
    }

def analyze_repository(repo_info: Dict, headers: Dict,
                      start_date: datetime.datetime, end_date: datetime.datetime,
                      skip_collect: bool = False, skip_git: bool = False) -> Dict:
    """分析单个仓库"""
    owner = repo_info["owner"]
    name = repo_info["name"]
    eco_id = repo_info["eco_id"]
    project_id_num = repo_info["project_id_num"]
    project_type = repo_info["project_type"]

    repo_start_time = time.time()  # 记录开始时间
    print(f"\n{'='*60}")
    print(f"开始分析仓库: {owner}/{name}")
    print(f"{'='*60}")

    # 获取仓库元数据
    print("  获取仓库元数据...")
    created_at, updated_at = get_repo_metadata(owner, name, headers)
    repo_info["created_at"] = created_at
    repo_info["updated_at"] = updated_at

    try:
        # 1. 收集API数据
        prs = []
        issues = []
        forks = []
        if not skip_collect:
            prs = collect_pr_data(owner, name, headers, start_date, end_date)
            issues = collect_issue_data(owner, name, headers, start_date, end_date)
            forks = collect_fork_data(owner, name, headers, start_date, end_date)
        else:
            print("  跳过API数据收集")

        # 2. 本地Git分析
        git_data = {}
        if not skip_git:
            git_data = analyze_git_repo(owner, name, start_date, end_date)
        else:
            print("  跳过Git分析")

        # 3. 计算月度数据
        months = calculate_monthly_metrics(prs, issues, forks, git_data, start_date, end_date, owner, name, headers)

        # 4. 计算总体指标
        results = calculate_overall_metrics(prs, issues, months, git_data, owner, name, headers)

        # 计算耗时
        repo_elapsed = round(time.time() - repo_start_time, 2)
        print(f"耗时：{repo_elapsed}秒")

        # 5. 添加仓库信息
        results["repository"] = {
            "owner": owner,
            "name": name,
            "full_name": f"{owner}/{name}",
            "eco_id": eco_id,
            "project_id_num": project_id_num,
            "project_type": project_type,
            "created_at": created_at.isoformat() if created_at else "",
            "updated_at": updated_at.isoformat() if updated_at else "",
            "analysis_time": datetime.datetime.now().isoformat(),
            "analysis_duration_seconds": repo_elapsed,  # 新增：分析耗时
            "time_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            }
        }

        print(f"\n✓ 完成分析: {owner}/{name}")
        print(f"{'='*60}")

        return results

    except Exception as e:
        print(f"✗ 分析仓库 {owner}/{name} 时出错: {e}")
        import traceback
        traceback.print_exc()
        return {
            "repository": {
                "owner": owner,
                "name": name,
                "full_name": f"{owner}/{name}",
                "eco_id": eco_id,
                "project_id_num": project_id_num,
                "project_type": project_type,
                "created_at": created_at.isoformat() if created_at else "",
                "updated_at": updated_at.isoformat() if updated_at else "",
                "error": str(e),
                "analysis_time": datetime.datetime.now().isoformat()
            },
            "error": True
        }

def export_to_csv(data: Dict, filename_prefix: str):
    """将仓库数据导出为CSV格式，分别保存月度数据和其他数据"""
    try:
        # 提取仓库基础信息
        repo_info = data.get("repository", {})
        owner = repo_info.get("owner", "")
        name = repo_info.get("name", "")
        full_name = repo_info.get("full_name", "")
        eco_id = repo_info.get("eco_id", "")
        project_id_num = repo_info.get("project_id_num", "")
        project_type = repo_info.get("project_type", "")
        created_at = repo_info.get("created_at", "")
        updated_at = repo_info.get("updated_at", "")

        formatted_created_at = format_to_year_month(created_at)
        formatted_updated_at = format_to_year_month(updated_at)

        # 提取总体指标
        overall = data.get("overall", {})

        # 提取月度数据
        monthly = data.get("monthly", {})

        # =============== 导出月度数据到单独文件 ===============
        if monthly:
            monthly_filename = f"{filename_prefix}_monthly.csv"
            with open(monthly_filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)

                # 写入月度数据表头
                headers = [
                    "模型生态",
                    "项目数值标识",
                    "项目类型",
                    "创建时间",
                    "最后更新时间",
                    "月度时间",
                    "月度活跃贡献者数",
                    "PR合并率 (%)",
                    "Issue解决周期 (天)(倒数)",
                    "评论数",
                    "反应数",
                    "每月评论和反应总数",
                    "RFC采纳率 (%)",
                    "提交量前10%开发者占比 (%)",
                    "月度非merge代码提交次数",
                    "月度发布新版本次数",
                    "月度Fork数",
                    "多作者PR次数",
                    "多作者PR比例 (%)"
                ]
                writer.writerow(headers)

                # 获取所有月份并按时间排序
                months = sorted(monthly.get("pr_counts", {}).keys())

                for month in months:
                    # 将月度时间格式化为"2023m11"格式
                    # month 格式是 "2023-11"，需要转换为 "2023m11"
                    formatted_month = format_to_year_month(month)

                    row = [
                        eco_id,                    # 模型生态
                        project_id_num,            # 项目数值标识
                        project_type,           # 项目类型
                        formatted_created_at,      # 创建时间
                        formatted_updated_at,      # 最后更新时间
                        formatted_month,           # 月度时间
                    ]

                    # 唯一贡献者数
                    row.append(monthly["unique_contributors"].get(month, 0))

                    # PR合并率
                    pr_merge_rate = monthly["pr_merge_rates"].get(month, 0)
                    row.append(f"{pr_merge_rate:.1f}")

                    # Issue解决周期
                    issue_resolution_days = monthly["avg_issue_resolution_days"].get(month, 0)
                    if issue_resolution_days > 0:
                        issue_resolution_inverse = 1 / issue_resolution_days
                        row.append(f"{issue_resolution_inverse:.2f}")
                    else:
                        row.append("0.00")

                    # 交互数据
                    interaction = monthly.get("interaction", {}).get(month, {})
                    row.append(interaction.get("comments", 0))  # 评论数
                    row.append(interaction.get("reactions", 0))  # 反应数
                    row.append(interaction.get("total", 0))  # 交互总数

                    # RFC采纳率
                    rfc_adoption_rate = monthly["rfc_adoption_rates"].get(month, 0)
                    row.append(f"{rfc_adoption_rate:.1f}")

                    # 提交量前10%开发者占比
                    top_10_percentage = monthly["top_10_developer_percentages"].get(month, 0)
                    row.append(f"{top_10_percentage:.1f}")

                    # 月度非merge代码提交次数
                    row.append(monthly["non_merge_commits"].get(month, 0))

                    # Tag和Fork数
                    row.append(monthly["tags"].get(month, 0))  # Tag数
                    row.append(monthly["forks"].get(month, 0))  # Fork数

                    # 多作者PR统计
                    # 多作者PR次数
                    row.append(monthly["multi_author_pr_counts"].get(month, 0))

                    # 多作者PR比例
                    multi_author_pr_rate = monthly["multi_author_pr_rates"].get(month, 0)
                    row.append(f"{multi_author_pr_rate:.1f}")

                    writer.writerow(row)

            print(f"  ✓ 月度数据已保存到: {monthly_filename}")

        # =============== 导出其他数据到单独文件 ===============
        other_filename = f"{filename_prefix}_other.csv"
        with open(other_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            # 写入仓库基本信息
            writer.writerow(["仓库基本信息"])
            writer.writerow(["所有者", owner])
            writer.writerow(["仓库名", name])
            writer.writerow(["完整名称", full_name])
            writer.writerow(["模型生态", eco_id])
            writer.writerow(["项目数值标识", project_id_num])
            writer.writerow(["项目类型", project_type])
            writer.writerow(["创建时间", formatted_created_at])  # 使用格式化后的创建时间
            writer.writerow(["最后更新时间", formatted_updated_at])  # 使用格式化后的更新时间
            writer.writerow([])

            # 写入总体指标
            writer.writerow(["总体指标"])
            writer.writerow(["指标", "值"])

            # PR合并率
            pr_merge_rate = overall.get("pr_merge_rate", 0)
            writer.writerow(["PR合并率 (%)", f"{pr_merge_rate:.1f}"])

            # Issue平均解决周期
            avg_issue_days = overall.get("avg_issue_resolution_days", 0)
            if avg_issue_days > 0:
                issue_inverse = 1 / avg_issue_days
                writer.writerow(["Issue平均解决周期 (天)(倒数)", f"{issue_inverse:.2f}"])
            else:
                writer.writerow(["Issue平均解决周期 (天)(倒数)", "0.00"])

            # RFC采纳率
            rfc_adoption_rate = overall.get("rfc_adoption_rate", 0)
            writer.writerow(["RFC采纳率 (%)", f"{rfc_adoption_rate:.1f}"])

            # RFC PR采纳率
            rfc_pr_adoption_rate = overall.get("rfc_pr_adoption_rate", 0)
            writer.writerow(["RFC PR采纳率 (%)", f"{rfc_pr_adoption_rate:.1f}"])

            # RFC Issue采纳率
            rfc_issue_adoption_rate = overall.get("rfc_issue_adoption_rate", 0)
            writer.writerow(["RFC Issue采纳率 (%)", f"{rfc_issue_adoption_rate:.1f}"])

            # 提交量前10%开发者占比
            top_10_percentage = overall.get("top_10_developer_commit_percentage", 0)
            writer.writerow(["提交量前10%开发者占比 (%)", f"{top_10_percentage:.1f}"])

            writer.writerow(["非merge提交数", overall.get("total_non_merge_commits", 0)])
            writer.writerow(["多作者PR次数", overall.get("multi_author_pr_count", 0)])

            # 多作者PR比例
            multi_author_pr_rate = overall.get("multi_author_pr_rate", 0)
            writer.writerow(["多作者PR比例 (%)", f"{multi_author_pr_rate:.1f}"])
            writer.writerow(["月度活跃贡献者数", overall.get("total_unique_contributors", 0)])
            writer.writerow(["月均贡献者数", overall.get("avg_monthly_contributors", 0)])
            writer.writerow(["Fork总数", overall.get("total_forks", 0)])
            writer.writerow(["Tag总数", overall.get("total_tags", 0)])
            writer.writerow(["PR总数", overall.get("total_prs", 0)])
            writer.writerow(["Issue总数", overall.get("total_issues", 0)])
            writer.writerow([])

            # 写入抽样统计（如果有）
            sample_stats = data.get("sample_stats", {})
            if sample_stats:
                writer.writerow(["抽样统计"])
                writer.writerow(["指标", "值"])
                writer.writerow(["PR作者抽样数量", sample_stats.get("pr_authors_sample_size", 0)])
                writer.writerow(["多作者PR数量(抽样中)", sample_stats.get("multi_author_prs_in_sample", 0)])
                writer.writerow([])

        print(f"  ✓ 其他数据已保存到: {other_filename}")
        return True

    except Exception as e:
        print(f"  ✗ 导出CSV失败: {e}")
        import traceback
        traceback.print_exc()
        return False

# ==================== 主函数 ====================
def main():
    """主函数"""
    project_start_time = time.time()  # 记录项目开始时间
    parser = argparse.ArgumentParser(description='GitHub仓库数据分析工具')
    parser.add_argument('--file', type=str, required=True,
                       help='包含仓库列表的文件路径 (格式: owner/name 或 owner name)')
    parser.add_argument('--months', type=str, required=True,
                       help='月份范围 (格式: YYYY.MM-YYYY.MM, 如: 2023.11-2025.06)')
    parser.add_argument('--token', type=str, required=True,
                       help='GitHub个人访问令牌')
    parser.add_argument('--output', type=str, default='github_analysis_results',
                       help='输出文件前缀 (默认: github_analysis_results)')
    parser.add_argument('--skip-collect', action='store_true',
                       help='跳过API数据收集')
    parser.add_argument('--skip-git', action='store_true',
                       help='跳过Git克隆分析')
    parser.add_argument('--debug', action='store_true',
                       help='启用调试模式')

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("GitHub仓库数据分析工具")
    print(f"{'='*60}")

    # 解析月份范围
    print(f"解析月份范围: {args.months}")
    start_date, end_date = parse_month_range(args.months)
    print(f"✓ 时间范围: {start_date.strftime('%Y-%m')} 至 {end_date.strftime('%Y-%m')}")

    # 读取仓库列表
    print(f"\n读取仓库列表文件: {args.file}")
    repos = read_repo_list(args.file)  # 现在返回的是字典列表

    # 设置请求头
    headers = {
        "Authorization": f"Bearer {args.token}",
        "Content-Type": "application/json",
    }

    # 检查速率限制
    if not check_rate_limit(headers):
        print("警告: API速率限制较低，可能无法完成所有分析")
        proceed = input("是否继续? (y/n): ")
        if proceed.lower() != 'y':
            print("退出分析")
            sys.exit(0)

    # 分析所有仓库（跳过重复项目）
    all_results = {}
    successful = 0
    failed = 0
    analyzed_keys = set()  # 记录已经分析过的仓库键

    # 首先去重，识别重复的仓库
    unique_repos = {}
    duplicate_repos = []

    print(f"检查重复仓库...")
    for repo_info in repos:
        repo_key = f"{repo_info['owner']}/{repo_info['name']}"
        if repo_key not in unique_repos:
            unique_repos[repo_key] = repo_info
        else:
            duplicate_repos.append(repo_key)

    if duplicate_repos:
        print(f"发现 {len(duplicate_repos)} 个重复仓库: {', '.join(sorted(set(duplicate_repos)))}")

    print(f"\n{'='*60}")
    print(f"开始分析 {len(unique_repos)} 个唯一仓库 (输入包含 {len(repos)} 个)")
    print(f"{'='*60}")

    repo_progress = ProgressTracker(total_steps=len(repos), description="仓库分析总进度")

    # 处理所有仓库（包括重复的）
    for i, repo_info in enumerate(repos, 1):
        owner = repo_info["owner"]
        name = repo_info["name"]
        repo_key = f"{owner}/{name}"
        result_key = f"input_{i:04d}_{owner}_{name}"
        print(f"\n[仓库 {i}/{len(repos)}]: {owner}/{name}")

        # 检查是否已经分析过此仓库
        if repo_key in analyzed_keys:
            print(f"  跳过重复仓库 {repo_key} 的API和Git分析，但保留自定义信息")

            original_result = all_results[repo_key]

            # 创建新的结果，复制数据但更新仓库信息
            result = {
                "overall": original_result.get("overall", {}),
                "monthly": original_result.get("monthly", {}),
                "sample_stats": original_result.get("sample_stats", {}),
                "repository": {
                    "owner": owner,
                    "name": name,
                    "full_name": f"{owner}/{name}",
                    "eco_id": repo_info.get('eco_id', ''),
                    "project_id_num": repo_info.get('project_id_num', ''),
                    "project_type": repo_info.get('project_type', ''),
                    "created_at": original_result.get("repository", {}).get("created_at", ""),
                    "updated_at": original_result.get("repository", {}).get("updated_at", ""),
                    "analysis_time": datetime.datetime.now().isoformat(),
                    "analysis_duration_seconds": 0,
                    "time_range": original_result.get("repository", {}).get("time_range", {}),
                    "duplicate_source": repo_key
                },
                "skipped": True
            }

            # 保存结果，使用输入顺序作为唯一标识
            all_results[result_key] = result

            successful += 1
            repo_progress.update(1, f"跳过 {owner}/{name}")
            continue

        # 分析仓库（新仓库）
        result = analyze_repository(repo_info, headers, start_date, end_date, args.skip_collect, args.skip_git)

        # 保存结果
        if result.get("error"):
            failed += 1
        else:
            successful += 1
            # 记录已分析过的仓库
            analyzed_keys.add(repo_key)

        all_results[repo_key] = result
        all_results[result_key] = result
        repo_progress.update(1, f"完成 {owner}/{name}")

        # 只对新分析的仓库生成文件
        if repo_key not in analyzed_keys or result.get("error"):
            # 跳过文件生成
            pass
        else:
            # 保存每个仓库的独立结果
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            # 为每个项目创建独立的文件夹
            project_dir = os.path.join(output_dir, f"{owner}_{name}")
            os.makedirs(project_dir, exist_ok=True)
            # 导出两个文件到项目文件夹：一个用于月度数据，一个用于其他数据
            output_prefix = os.path.join(project_dir, f"{args.output}_{owner}_{name}")
            export_to_csv(result, output_prefix)

        # 每个仓库完成后检查剩余API点数
        print(f"  仓库 {owner}/{name} 完成，检查剩余API点数:")
        check_rate_limit(headers)

        # 在仓库之间添加延迟，避免触发速率限制
        if i < len(repos):
            wait_time = 3
            print(f"  等待{wait_time}秒后继续下一个仓库...")
            time.sleep(wait_time)

    repo_progress.complete()

    # 计算项目总耗时
    project_elapsed = round(time.time() - project_start_time, 2)
    print(f"总耗时：{project_elapsed}秒")

    # 保存汇总结果
    print(f"\n保存汇总结果...")

    summary = {
        "metadata": {
            "total_repos": len(repos),
            "successful": successful,
            "failed": failed,
            "analysis_time": datetime.datetime.now().isoformat(),
            "analysis_duration_seconds": project_elapsed,
            "time_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "skip_collect": args.skip_collect,
            "skip_git": args.skip_git
        },
        "repositories": all_results
    }

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    csv_summary_file = os.path.join(output_dir, f"{args.output}_summary.csv")
    try:
        with open(csv_summary_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            # 写入元数据
            writer.writerow(["分析汇总报告"])
            writer.writerow(["分析时间", summary["metadata"]["analysis_time"]])
            writer.writerow(["时间范围", f"{start_date.strftime('%Y-%m')} 至 {end_date.strftime('%Y-%m')}"])
            writer.writerow(["总仓库数", summary["metadata"]["total_repos"]])
            writer.writerow(["成功分析", summary["metadata"]["successful"]])
            writer.writerow(["失败分析", summary["metadata"]["failed"]])
            writer.writerow(["总耗时(秒)", summary["metadata"]["analysis_duration_seconds"]])
            writer.writerow(["跳过API数据收集", summary["metadata"]["skip_collect"]])
            writer.writerow(["跳过Git分析", summary["metadata"]["skip_git"]])
            writer.writerow([])

            # 写入各仓库总体指标 - 严格按照输入顺序
            writer.writerow(["各仓库总体指标"])
            headers = ["仓库", "模型生态", "项目数值标识", "项目类型", "PR合并率 (%)", "Issue解决周期 (天)", "RFC采纳率 (%)",
                    "前10%开发者占比 (%)", "非merge代码提交次数", "多作者PR次数", "多作者PR比例 (%)",
                    "月度活跃贡献者数", "Fork总数", "Tag总数", "PR总数", "Issue总数", "创建时间", "最后更新时间"]
            writer.writerow(headers)

            # 严格按照输入顺序写入
            for i, repo_info in enumerate(repos, 1):
                owner = repo_info["owner"]
                name = repo_info["name"]
                repo_key = f"{owner}/{name}"

                # 查找对应的结果
                result_key = f"input_{i:04d}_{owner}_{name}"
                result = all_results[result_key]

                if result:
                    overall = result.get("overall", {})
                    repo = result.get("repository", {})
                    row = [repo_key]
                    row.append(repo.get("eco_id", ""))
                    row.append(repo.get("project_id_num", ""))
                    row.append(repo.get("project_type", ""))

                    if result.get("error"):
                        # 错误情况填0
                        row.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                    else:
                        row.append(overall.get("pr_merge_rate", 0))
                        row.append(overall.get("avg_issue_resolution_days", 0))
                        row.append(overall.get("rfc_adoption_rate", 0))
                        row.append(overall.get("top_10_developer_commit_percentage", 0))
                        row.append(overall.get("total_non_merge_commits", 0))
                        row.append(overall.get("multi_author_pr_count", 0))
                        row.append(overall.get("multi_author_pr_rate", 0))
                        row.append(overall.get("total_unique_contributors", 0))
                        row.append(overall.get("total_forks", 0))
                        row.append(overall.get("total_tags", 0))
                        row.append(overall.get("total_prs", 0))
                        row.append(overall.get("total_issues", 0))

                    row.append(repo.get("created_at", ""))
                    row.append(repo.get("updated_at", ""))
                    writer.writerow(row)

        print(f"✓ CSV汇总结果已保存到: {csv_summary_file}")

    except Exception as e:
        print(f"✗ 保存CSV汇总结果失败: {e}")

    # 合并各仓库的月度数据到一个大表
    if successful > 0:
        print(f"\n{'='*60}")
        print("合并各仓库月度数据...")
        print(f"{'='*60}")

        output_dir = "output"
        github_data_file = os.path.join(output_dir, "github_data.csv")

        try:
            all_monthly_data = []

            # 修改：按照输入顺序遍历仓库
            for i, repo_info in enumerate(repos, 1):
                owner = repo_info["owner"]
                name = repo_info["name"]
                repo_key = f"{owner}/{name}"
                result_key = f"input_{i:04d}_{owner}_{name}"

                # 查找对应的结果
                result = all_results.get(result_key)

                if result and not result.get("error"):
                    # 从结果中获取仓库信息
                    repo_info_data = result.get("repository", {})
                    eco_id = repo_info_data.get('eco_id', '')
                    project_id_num = repo_info_data.get('project_id_num', '')
                    project_type = repo_info_data.get('project_type', '')
                    created_at = repo_info_data.get("created_at", "")
                    updated_at = repo_info_data.get("updated_at", "")

                    # 直接从内存中的monthly数据生成
                    monthly_data = result.get("monthly", {})
                    if monthly_data:
                        # 获取所有月份
                        months = sorted(monthly_data.get("pr_counts", {}).keys())

                        for month in months:
                            # 将月度时间格式化为"2023m11"格式
                            formatted_month = format_to_year_month(month)

                            # 构建数据行
                            row = [
                                f"{owner}/{name}",  # 项目名称
                                eco_id,
                                project_id_num,
                                project_type,
                                created_at,  # 创建时间
                                updated_at,  # 最后更新时间
                                formatted_month,  # 月份
                            ]

                            # 添加月度数据
                            # 1. 月度活跃贡献者数
                            row.append(monthly_data.get("unique_contributors", {}).get(month, 0))

                            # 2. PR合并率
                            pr_merge_rate = monthly_data.get("pr_merge_rates", {}).get(month, 0)
                            row.append(f"{pr_merge_rate:.1f}")

                            # 3. Issue解决周期（倒数）
                            issue_days = monthly_data.get("avg_issue_resolution_days", {}).get(month, 0)
                            if issue_days > 0:
                                issue_inverse = 1 / issue_days
                                row.append(f"{issue_inverse:.2f}")
                            else:
                                row.append("0.00")

                            # 4. 交互数据总数
                            interaction = monthly_data.get("interaction", {}).get(month, {})
                            row.append(interaction.get("total", 0))

                            # 5. RFC采纳率
                            rfc_rate = monthly_data.get("rfc_adoption_rates", {}).get(month, 0)
                            row.append(f"{rfc_rate:.1f}")

                            # 6. 前10%开发者占比
                            top10_rate = monthly_data.get("top_10_developer_percentages", {}).get(month, 0)
                            row.append(f"{top10_rate:.1f}")

                            # 7. 非merge提交数
                            row.append(monthly_data.get("non_merge_commits", {}).get(month, 0))

                            # 8. Tag数
                            row.append(monthly_data.get("tags", {}).get(month, 0))

                            # 9. Fork数
                            row.append(monthly_data.get("forks", {}).get(month, 0))

                            # 10. 多作者PR次数
                            row.append(monthly_data.get("multi_author_pr_counts", {}).get(month, 0))

                            # 11. 多作者PR比例
                            multi_author_rate = monthly_data.get("multi_author_pr_rates", {}).get(month, 0)
                            row.append(f"{multi_author_rate:.1f}")

                            all_monthly_data.append(row)

            # 如果有月度数据，写入合并文件
            if all_monthly_data:
                # 准备新的表头格式
                # 第一行：英文表头
                header_en = [
                    "repo", "eco_id", "project_id_num", "project_type",
                    "created_at", "updated_at", "month_date", "monthly_active_contributors", "pr_merge_rate",
                    "issue_resolve_cycle", "comment_react_total", "rfc_adopt_rate",
                    "top10_dev_ratio", "nonmerge_commit_month", "new_version_month",
                    "fork_num", "cross_org_collab", "contributor_network_density"
                ]

                # 第二行：中文表头
                header_cn = [
                    "项目名称", "模型生态", "项目数值标识", "项目类型",
                    "创建时间", "最后更新时间", "月份", "月度活跃贡献者数",
                    "月度PR合并率 (%)", "月度Issue解决周期 (天)(倒数)", "月度评论和反应总数",
                    "月度RFC采纳率 (%)", "月度提交量前10%开发者占比 (%)", "月度非merge代码提交次数",
                    "月度发布新版本次数", "月度Fork数", "月度多作者PR次数", "月度多作者PR比例 (%)"
                ]

                # 重新处理数据以匹配新的表头结构
                processed_data = []
                for row in all_monthly_data:
                    # 原始数据列对应关系:
                    # 0: 项目名称, 1: 模型生态, 2: 项目数值标识, 3: 项目类型,
                    # 4: 创建时间, 5: 最后更新时间, 6: 月度时间, 7: 月度活跃贡献者数,
                    # 8: PR合并率 (%), 9: Issue解决周期 (天)(倒数), 10: 每月评论和反应总数,
                    # 11: RFC采纳率 (%), 12: 提交量前10%开发者占比 (%),
                    # 13: 月度非merge代码提交次数, 14: 月度发布新版本次数, 15: 月度Fork数,
                    # 16: 多作者PR次数, 17: 多作者PR比例 (%)

                    # 创建新的数据行，按照要求的顺序
                    new_row = [
                        row[0],  # repo (项目名称)
                        row[1],  # eco_id (模型生态)
                        row[2],  # project_id_num (项目数值标识)
                        row[3],  # project_type (项目类型)
                        format_to_year_month(row[4]),  # created_at (创建时间)
                        format_to_year_month(row[5]),  # updated_at (最后更新时间)
                        row[6],  # month_date (月份)
                        row[7],  # 月度活跃贡献者数
                        row[8],  # pr_merge_rate (月度PR合并率 %)
                        row[9],  # issue_resolve_cycle (月度Issue解决周期倒数)
                        row[10], # comment_react_total (月度评论和反应总数)
                        row[11], # rfc_adopt_rate (月度RFC采纳率 %)
                        row[12], # top10_dev_ratio (月度提交量前10%开发者占比 %)
                        row[13], # nonmerge_commit_month (月度非merge代码提交次数)
                        row[14], # new_version_month (月度发布新版本次数)
                        row[15], # fork_num (月度Fork数)
                        row[16], # cross_org_collab (月度多作者PR次数)
                        row[17]  # contributor_network_density (月度多作者PR比例 %)
                    ]
                    processed_data.append(new_row)

                with open(github_data_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)

                    # 写入第一行：英文表头
                    writer.writerow(header_en)

                    # 写入第二行：中文表头
                    writer.writerow(header_cn)

                    # 写入所有数据（从第三行开始）
                    writer.writerows(processed_data)

                print(f"  ✓ 合并完成! 共合并 {len(all_monthly_data)} 行数据")
                print(f"  ✓ 合并数据已保存到: {github_data_file}")
            else:
                print("  ✗ 没有找到可合并的月度数据文件")

        except Exception as e:
            print(f"  ✗ 合并月度数据失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("  ✗ 跳过月度数据合并 (格式不是csv或没有成功分析)")

    # 输出摘要报告
    print(f"\n{'='*60}")
    print("分析完成!")
    print(f"{'='*60}")
    print(f"输入仓库数: {len(repos)}")
    print(f"唯一仓库数: {len(unique_repos)}")
    print(f"重复仓库数: {len(repos) - len(unique_repos)}")
    print(f"成功分析: {successful} (包含复用)")
    print(f"实际分析: {len(analyzed_keys)}")
    print(f"失败分析: {failed}")
    print(f"月份范围: {start_date.strftime('%Y-%m')} 至 {end_date.strftime('%Y-%m')}")
    print(f"项目总耗时: {project_elapsed:.1f}秒")
    print(f"独立结果: {args.output}_<owner>_<name> 在各自项目文件夹下")
    print(f"汇总结果: {args.output}_summary.csv")
    print(f"合并数据: github_data.csv (按输入顺序输出)")  # 修改提示信息

    print(f"\n{'='*60}")
    print("所有任务完成!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
#!/bin/bash

################################################################################
# Automated Setup Script for swavlamban-waf-ml
# This script automates the setup and initialization of the project
# Created: 2025-12-25
################################################################################

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

################################################################################
# Helper Functions
################################################################################

print_header() {
    echo -e "\n${BLUE}===================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}===================================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

################################################################################
# Check Prerequisites
################################################################################

check_prerequisites() {
    print_header "Checking Prerequisites"
    
    # Check if running on Unix-like system
    if [[ ! "$OSTYPE" == "linux-gnu"* ]] && [[ ! "$OSTYPE" == "darwin"* ]]; then
        print_error "This script is designed for Linux/macOS systems"
        exit 1
    fi
    print_success "Operating System: $OSTYPE"
    
    # Check Python installation
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed"
        exit 1
    fi
    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    print_success "Python 3 found: $PYTHON_VERSION"
    
    # Check pip installation
    if ! command -v pip3 &> /dev/null; then
        print_error "pip3 is not installed"
        exit 1
    fi
    print_success "pip3 found"
    
    # Check Git installation
    if ! command -v git &> /dev/null; then
        print_error "Git is not installed"
        exit 1
    fi
    GIT_VERSION=$(git --version)
    print_success "$GIT_VERSION"
}

################################################################################
# Setup Virtual Environment
################################################################################

setup_virtual_env() {
    print_header "Setting up Python Virtual Environment"
    
    if [ -d "venv" ]; then
        print_warning "Virtual environment already exists"
        read -p "Do you want to recreate it? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf venv
            print_info "Removed existing virtual environment"
        else
            print_info "Using existing virtual environment"
            return
        fi
    fi
    
    python3 -m venv venv
    print_success "Virtual environment created"
    
    # Activate virtual environment
    source venv/bin/activate
    print_success "Virtual environment activated"
    
    # Upgrade pip
    pip3 install --upgrade pip setuptools wheel
    print_success "pip, setuptools, and wheel upgraded"
}

################################################################################
# Install Dependencies
################################################################################

install_dependencies() {
    print_header "Installing Project Dependencies"
    
    if [ -f "requirements.txt" ]; then
        pip3 install -r requirements.txt
        print_success "Dependencies from requirements.txt installed"
    else
        print_warning "requirements.txt not found"
    fi
    
    if [ -f "requirements-dev.txt" ]; then
        pip3 install -r requirements-dev.txt
        print_success "Development dependencies installed"
    fi
}

################################################################################
# Initialize Project Structure
################################################################################

initialize_project_structure() {
    print_header "Initializing Project Structure"
    
    # Create necessary directories
    directories=("data" "models" "logs" "notebooks" "tests" "src")
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            print_success "Created directory: $dir"
        else
            print_info "Directory already exists: $dir"
        fi
    done
}

################################################################################
# Setup Configuration Files
################################################################################

setup_config() {
    print_header "Setting up Configuration Files"
    
    if [ ! -f ".env.example" ] && [ ! -f ".env" ]; then
        print_warning "No .env configuration found"
        print_info "Please create .env file with required environment variables"
    fi
    
    if [ -f ".env.example" ] && [ ! -f ".env" ]; then
        cp .env.example .env
        print_success "Created .env from .env.example"
        print_warning "Please update .env with your configuration"
    fi
}

################################################################################
# Run Tests
################################################################################

run_tests() {
    print_header "Running Tests"
    
    if [ -d "tests" ] && [ "$(ls -A tests)" ]; then
        if command -v pytest &> /dev/null; then
            pytest tests/ -v
            print_success "Tests completed"
        else
            print_warning "pytest not installed, skipping tests"
        fi
    else
        print_info "No tests found"
    fi
}

################################################################################
# Display Summary
################################################################################

display_summary() {
    print_header "Setup Complete!"
    
    echo -e "${GREEN}The project has been successfully set up.${NC}\n"
    echo "Next steps:"
    echo "  1. Activate the virtual environment: source venv/bin/activate"
    echo "  2. Update .env with your configuration"
    echo "  3. Review the project structure in README.md"
    echo "  4. Start developing!"
    echo ""
}

################################################################################
# Main Execution
################################################################################

main() {
    echo -e "${BLUE}"
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║      swavlamban-waf-ml - Automated Setup Script           ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    check_prerequisites
    setup_virtual_env
    install_dependencies
    initialize_project_structure
    setup_config
    run_tests
    display_summary
}

# Execute main function
main "$@"
